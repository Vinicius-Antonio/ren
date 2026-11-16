from __future__ import annotations

import time
from pathlib import Path

from src.audio.vad import NeuralVAD, compute_fft_bands
from src.config import PROJECT_ROOT
from src.core.router import EdgeRouter, SlowLoopManager
from src.gateway.ws_server import GatewayServer, GatewayStatus
from src.memory.sqlite_store import EpisodicMemoryStore
from src.system import briefing, email_client, pc_actions, timers
from src.vision.screen import ScreenCaptureWorker, capture_screen_jpeg, is_screen_request


def test_screen_request_phrases_and_jpeg_capture(monkeypatch: object) -> None:
    from PIL import Image, ImageGrab

    assert is_screen_request("Hermes, olha a minha tela e me diz o que acha")
    assert is_screen_request("Me explica esse erro aqui")
    assert is_screen_request("O que tem na tela agora?")
    assert is_screen_request("Qual item eu faço nessa partida?")
    assert not is_screen_request("Qual a previsão do tempo para amanhã?")

    monkeypatch.setattr(
        ImageGrab,
        "grab",
        lambda all_screens=False: Image.new("RGB", (1920, 1080), color=(20, 40, 80)),
    )  # type: ignore[attr-defined]
    jpeg = capture_screen_jpeg(max_width=1280, quality=60, max_height=720)
    assert jpeg.startswith(b"\xff\xd8")  # Assinatura JPEG válida
    assert len(jpeg) > 500

    # Verifica fila drop-frame (maxsize=1) do ScreenCaptureWorker
    worker = ScreenCaptureWorker(interval_sec=1.0)
    worker.put_latest_frame(b"frame-1")
    worker.put_latest_frame(b"frame-2")
    assert worker._frame_queue.qsize() == 1
    assert worker._frame_queue.get_nowait() == b"frame-2"


def test_gateway_contracts_and_fft_telemetry() -> None:
    gw = GatewayServer(enabled=False)
    st = gw.emit_state_changed(GatewayStatus.LISTENING)
    assert st == {"type": "STATE_CHANGED", "status": "LISTENING"}
    assert gw.current_status is GatewayStatus.LISTENING

    pcm_chunk = b"\x10\x08" * 512
    bands = compute_fft_bands(pcm_chunk, num_bands=16)
    assert len(bands) == 16
    assert all(0.0 <= b <= 1.0 for b in bands)

    lvl1 = gw.emit_audio_chunk_telemetry(pcm_chunk, force=True)
    assert lvl1 is not None
    assert lvl1["type"] == "AUDIO_LEVEL"
    assert len(lvl1["frequencies"]) == 16

    # Throttling de 50ms impede inundação do WebSocket
    assert gw.emit_audio_chunk_telemetry(pcm_chunk, force=False) is None

    w = gw.emit_ui_widget("executive_briefing", {"city": "Santa Mariana"})
    assert w == {
        "type": "UI_WIDGET",
        "widget": "executive_briefing",
        "payload": {"city": "Santa Mariana"},
    }


def test_edge_router_and_slow_loop(monkeypatch: object) -> None:
    monkeypatch.setattr(pc_actions, "_press_vk", lambda vk, times=1: None)  # type: ignore[attr-defined]
    gw = GatewayServer(enabled=False)
    router = EdgeRouter(gateway=gw)

    match = router.try_dispatch("Hermes, aumenta o volume por favor")
    assert match.matched is True
    assert match.intent == "volume_up"
    # Não dispara duas vezes no mesmo turno
    assert router.try_dispatch("Hermes, aumenta o volume por favor").matched is False
    router.reset_turn()
    assert router.try_dispatch("pula a música").intent == "next"

    completed: list[tuple[str, str]] = []
    slow = SlowLoopManager(
        gateway=gw, on_complete_callback=lambda name, res: completed.append((name, res))
    )
    info = slow.submit_task("relatorio_pesado", lambda: "Relatório concluído")
    assert info["status"] == "STARTED"
    for _ in range(20):
        if completed:
            break
        time.sleep(0.02)
    slow.shutdown()
    assert completed == [("relatorio_pesado", "Relatório concluído")]


def test_sqlite_episodic_memory_and_consolidation(tmp_path: Path) -> None:
    store = EpisodicMemoryStore(db_path=tmp_path / "ren_memory.db")
    store.add_episodic_memory("Mora em Santa Mariana, PR", categoria="preferencia")
    store.add_episodic_memory("Mora em Santa Mariana, PR", categoria="preferencia")
    mems = store.list_episodic_memories()
    assert len(mems) == 1
    assert mems[0]["categoria"] == "preferencia"

    extracted = store.consolidate_session(
        turns=[
            {
                "user": "Eu estou trabalhando no projeto Hermes 2.0 em Python",
                "assistant": "Excelente projeto!",
            }
        ]
    )
    assert any(item["categoria"] == "projeto" for item in extracted)
    assert len(store.list_session_history()) == 1
    assert store.delete_episodic_memory("Santa Mariana") == 1


def test_neural_vad_silero_onnx_512_samples() -> None:
    model_path = PROJECT_ROOT / "models" / "silero_vad.onnx"
    if not model_path.is_file():
        return
    vad = NeuralVAD(model_path, threshold=0.5)
    # 512 amostras de silêncio = 1024 bytes
    prob = vad.speech_probability(b"\x00\x00" * 512)
    assert 0.0 <= prob < 0.2
    assert vad.is_speech(b"\x00\x00" * 512) is False


def test_pc_actions_spotify_media_and_status(monkeypatch: object) -> None:
    started_uris: list[str] = []
    pressed_keys: list[tuple[int, int]] = []

    monkeypatch.setattr(pc_actions, "_start_uri", started_uris.append)  # type: ignore[attr-defined]
    monkeypatch.setattr(pc_actions, "_auto_play_spotify_search", lambda: None)  # type: ignore[attr-defined]
    monkeypatch.setattr(
        pc_actions,
        "_press_vk",
        lambda vk, times=1: pressed_keys.append((vk, times)),
    )  # type: ignore[attr-defined]

    res = pc_actions.play_music("Bohemian Rhapsody Queen", platform="spotify")
    assert "Bohemian Rhapsody" in res
    assert started_uris and started_uris[0].startswith("spotify:search:")

    assert "Play/Pause" in pc_actions.control_media("play_pause")
    assert "próxima" in pc_actions.control_media("next")
    assert "Volume aumentado" in pc_actions.control_media("volume_up", steps=3)
    assert len(pressed_keys) == 3

    status = pc_actions.get_pc_status()
    assert "Memória RAM:" in status
    assert "Disco C:" in status


def test_executive_briefing_combines_all_sources(monkeypatch: object) -> None:
    monkeypatch.setattr(
        briefing, "fetch_weather_summary", lambda city="": "Clima em SP: 24°C, limpo."
    )  # type: ignore[attr-defined]
    monkeypatch.setattr(
        briefing, "fetch_market_quotes", lambda: "Dólar a R$ 5.50, Bitcoin a R$ 425,000."
    )  # type: ignore[attr-defined]
    monkeypatch.setattr(
        briefing, "fetch_top_brazil_news", lambda limit=3: "Manchetes: 1) Notícia A"
    )  # type: ignore[attr-defined]

    text = briefing.build_executive_briefing(memories=["Revisar PR do Hermes"])
    assert "Clima em SP: 24°C" in text
    assert "Dólar a R$ 5.50" in text
    assert "Manchetes: 1) Notícia A" in text
    assert "Revisar PR do Hermes" in text
    assert "Memória RAM:" in text


def test_timers_voice_notes_and_email_client(tmp_path: object) -> None:
    from src.gemini.knowledge import KnowledgeManager

    p = Path(str(tmp_path))
    fired: list[str] = []
    tm = timers.TimerManager(on_fire_callback=fired.append)
    res = tm.set_timer(1, " Café pronto ")
    assert "Café pronto" in res
    assert "Timers ativos" in tm.list_timers()
    tm.cancel_all()
    assert "Nenhum timer" in tm.list_timers()

    km = KnowledgeManager(memory_file=p / "mem.json", knowledge_dir=p / "knowledge", timer_manager=tm)
    note_msg = km.create_voice_note("Ideias Projeto", "Testar integração com IMAP")
    assert "ideias_projeto.md" in note_msg
    assert "integração com IMAP" in km.search_local_knowledge("IMAP")

    assert "EMAIL_ADDRESS" in email_client.read_recent_emails()
