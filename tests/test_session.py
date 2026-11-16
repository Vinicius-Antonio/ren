from __future__ import annotations

import asyncio
from typing import Any, List

import pytest

import run as run_module
import src.gemini.session as session_mod
from src.audio.stream_manager import AudioStreamManager
from src.gemini.prompts import SYSTEM_INSTRUCTION, is_farewell, is_leave_call_command
from src.gemini.session import GeminiLiveSession, SessionEndReason
_REAL_BUILD_CONFIG = GeminiLiveSession.build_config
from tests.conftest import (
    FakeGenaiClient,
    FakeLiveSession,
    FakePyAudioModule,
    FakeStream,
    audio_message,
    interrupted_message,
    make_settings,
    pcm,
    transcription_message,
    turn_complete_message,
)


@pytest.fixture(autouse=True)
def fast_timing(monkeypatch: pytest.MonkeyPatch) -> List[bytes]:
    """Acelera o polling e evita depender do SDK real para config/envio."""
    monkeypatch.setattr(session_mod, "_POLL_SEC", 0.01)
    monkeypatch.setattr(session_mod, "_SPEAKING_TAIL_SEC", 0.05)
    sent: List[bytes] = []

    async def fake_send(self: Any, session: Any, data: bytes) -> None:
        sent.append(data)

    monkeypatch.setattr(GeminiLiveSession, "_send_audio", fake_send)
    monkeypatch.setattr(GeminiLiveSession, "build_config", lambda self: {"cfg": True})
    return sent


def _make(client: FakeGenaiClient, **settings: Any) -> tuple:
    pa = FakePyAudioModule()
    sess = GeminiLiveSession(make_settings(**settings), AudioStreamManager(pa), client)
    return sess, pa


def _run(sess: GeminiLiveSession) -> SessionEndReason:
    return asyncio.run(asyncio.wait_for(sess.run(), timeout=5))


# ------------------------------------------------------------------ prompts
@pytest.mark.parametrize(
    "text",
    ["Tchau!", "ok, até logo", "Até a próxima, Hermes", "pode desligar", "Valeu Hermes"],
)
def test_farewell_detected(text: str) -> None:
    assert is_farewell(text)


@pytest.mark.parametrize("text", ["qual a previsão do tempo?", "me conta uma piada", ""])
def test_farewell_not_detected(text: str) -> None:
    assert not is_farewell(text)


@pytest.mark.parametrize("text", ["Ren, sai da call", "desconecta do Discord", "saia do canal"])
def test_leave_call_command_detected(text: str) -> None:
    assert is_leave_call_command(text)


def test_unrelated_farewell_does_not_leave_call() -> None:
    assert not is_leave_call_command("Tchau, até logo")


def test_system_instruction_forbids_markdown() -> None:
    assert "Markdown" in SYSTEM_INSTRUCTION
    assert "busca do Google" in SYSTEM_INSTRUCTION
    assert "Personalidade e Tom" in SYSTEM_INSTRUCTION
    assert "Exemplo 1" in SYSTEM_INSTRUCTION


def test_is_addressed_to_hermes_phonetic_variants() -> None:
    from src.gemini.prompts import is_addressed_to_hermes

    assert is_addressed_to_hermes("Ei, eu queria saber por que não está respondendo")
    assert is_addressed_to_hermes("Rein, olha isso aqui")
    assert is_addressed_to_hermes("Hey, me ajuda com isso")
    assert is_addressed_to_hermes("Pain, qual é a resposta?")
    assert is_addressed_to_hermes("Rain, o que acha?")
    assert is_addressed_to_hermes("Hein, o que você acha?")
    assert is_addressed_to_hermes("Ren, salva uma nota")
    assert not is_addressed_to_hermes("Bora jogar outra partida agora")


def test_build_system_instruction_proactive_audio_discord() -> None:
    from src.gemini.prompts import build_system_instruction

    with_call = build_system_instruction(in_discord_call=True)
    assert "Reforço de Áudio Proativo (Call do Discord)" in with_call
    assert "Hermes" in with_call

    without_call = build_system_instruction(in_discord_call=False)
    assert "Reforço de Áudio Proativo (Call do Discord)" not in without_call


# ------------------------------------------------------------------ session
def test_silence_ends_session_and_closes_streams(fast_timing: List[bytes]) -> None:
    client = FakeGenaiClient()
    sess, pa = _make(client)
    assert _run(sess) is SessionEndReason.SILENCE
    assert client.connected and client.disconnected
    assert client.connect_kwargs["model"] == "test-model"
    assert fast_timing, "áudio do microfone deveria ter sido enviado"
    assert len(pa.streams) == 2 and all(s.closed for s in pa.streams)


def test_plays_audio_then_ends_on_farewell() -> None:
    live = FakeLiveSession(
        turns=[
            [audio_message(b"\x01\x00" * 100), turn_complete_message()],
            [
                transcription_message("Tchau, "),
                transcription_message("Hermes"),
                audio_message(b"\x02\x00" * 100),
                turn_complete_message(),
            ],
        ]
    )
    sess, pa = _make(FakeGenaiClient(live), vad_silence_timeout_sec=3.0)
    assert _run(sess) is SessionEndReason.FAREWELL
    out_stream = pa.streams[1]
    assert out_stream.written == [b"\x01\x00" * 100, b"\x02\x00" * 100]
    assert all(s.closed for s in pa.streams)


def test_connect_error_returns_gracefully() -> None:
    sess, pa = _make(FakeGenaiClient(connect_error=ConnectionError("sem rede")))
    assert _run(sess) is SessionEndReason.ERROR
    assert all(s.closed for s in pa.streams)


def test_receive_error_mid_session_returns_gracefully() -> None:
    live = FakeLiveSession(receive_error=RuntimeError("websocket caiu"))
    sess, pa = _make(FakeGenaiClient(live), vad_silence_timeout_sec=3.0)
    assert _run(sess) is SessionEndReason.ERROR
    assert all(s.closed for s in pa.streams)


def test_audio_device_error_returns_gracefully() -> None:
    client = FakeGenaiClient()
    sess, pa = _make(client)
    pa.fail_open = True
    assert _run(sess) is SessionEndReason.ERROR
    assert not client.connected  # nem chega a abrir conexão


def test_session_can_run_twice() -> None:
    client = FakeGenaiClient()
    sess, _ = _make(client)
    assert _run(sess) is SessionEndReason.SILENCE
    assert _run(sess) is SessionEndReason.SILENCE


def test_interrupted_drains_queue() -> None:
    sess, _ = _make(FakeGenaiClient())

    async def go() -> None:
        sess._reset_state()
        sess._handle_message(audio_message(b"\x01\x00"))
        sess._handle_message(audio_message(b"\x02\x00"))
        assert sess._audio_queue.qsize() == 2
        assert sess.assistant_is_speaking()
        sess._handle_message(interrupted_message())
        assert sess._audio_queue.empty()

    asyncio.run(go())


def test_call_server_interruption_clears_old_reply() -> None:
    sess, _ = _make(FakeGenaiClient())
    source = type("Source", (), {"cleared": False, "clear": lambda self: setattr(self, "cleared", True)})()
    sess._discord_bot = type("CallBot", (), {"is_in_call": True, "_audio_source": source})()
    sess._reset_state()
    sess._audio_queue.put_nowait(b"reply")
    sess._handle_message(interrupted_message())
    assert sess._audio_queue.empty()
    assert source.cleared


def test_requested_call_exit_is_distinct_from_standby() -> None:
    sess, _ = _make(FakeGenaiClient())
    sess._call_only = True
    sess._discord_bot = type("Bot", (), {"is_in_call": True})()  # type: ignore[assignment]
    sess._reset_state()
    sess._handle_message(transcription_message("Ren, sai da call"))
    assert sess._leave_call_requested
    assert sess._farewell_requested


def test_call_reads_only_the_member_who_called(monkeypatch: pytest.MonkeyPatch) -> None:
    class Bot:
        is_in_call = True

        def __init__(self) -> None:
            self.requested: List[int] = []

        def get_speaker_chunk(self, speaker_id: int, size: int) -> bytes:
            self.requested.append(speaker_id)
            return pcm(111 if speaker_id == 101 else 222)

    bot = Bot()
    sess, _ = _make(FakeGenaiClient())
    sess._discord_bot = bot  # type: ignore[assignment]
    sess._call_only = True
    sess.set_call_speaker(202)
    async def no_wait(_: float) -> None:
        return None

    monkeypatch.setattr(session_mod.asyncio, "sleep", no_wait)
    coroutine = sess._read_input(None, None)
    with pytest.raises(StopIteration) as finished:
        coroutine.send(None)
    data = finished.value.value
    assert bot.requested == [202]
    assert data == pcm(222)


def test_call_prompt_names_the_caller_without_assuming_creator() -> None:
    sess, _ = _make(FakeGenaiClient(), discord_user_id=101)
    sess._discord_bot = type("Bot", (), {
        "is_in_call": True, "active_channel": "Geral", "active_guild": "Teste",
        "speaker_name": lambda self, uid: "Bruno",
    })()  # type: ignore[assignment]
    sess.set_call_speaker(202)
    config = _REAL_BUILD_CONFIG(sess)
    prompt = config.system_instruction.parts[0].text
    assert "Bruno" in prompt
    assert "ID Discord 202" in prompt
    assert "Este ID não é o do Vinícius" in prompt


def test_gemini_38_call_config_uses_functions_without_unavailable_search() -> None:
    sess, _ = _make(FakeGenaiClient(), gemini_model="gemini-3.8-live")
    config = _REAL_BUILD_CONFIG(sess)
    assert config.speech_config.language_code is None
    assert not any(tool.google_search is not None for tool in config.tools)
    assert any(tool.function_declarations for tool in config.tools)


def test_default_live_config_keeps_search() -> None:
    sess, _ = _make(FakeGenaiClient())
    config = _REAL_BUILD_CONFIG(sess)
    assert config.speech_config.language_code == "pt-BR"
    assert any(tool.google_search is not None for tool in config.tools)


def test_call_orchestrator_passes_wake_author_to_session(monkeypatch: pytest.MonkeyPatch) -> None:
    class Detector:
        label = "Ren"
        wake_speaker_id = 202
        started = False
        stopped = False

        def wait_for_wake_word(self) -> None:
            pass

        def has_pending_call_speaker(self) -> bool:
            return False

        def start_call_monitor(self) -> None:
            self.started = True

        def stop_call_monitor(self) -> None:
            self.stopped = True

    class Session:
        speaker_id = None

        def set_call_speaker(self, speaker_id: int) -> None:
            self.speaker_id = speaker_id

        def set_call_handoff_checker(self, checker: Any) -> None:
            self.checker = checker

        async def run(self) -> SessionEndReason:
            return SessionEndReason.SILENCE

    detector = Detector()
    session = Session()
    def run_without_loop(coro: Any) -> SessionEndReason:
        try:
            coro.send(None)
        except StopIteration as finished:
            return finished.value
        raise AssertionError("run deveria terminar sem aguardar")

    monkeypatch.setattr(run_module.asyncio, "run", run_without_loop)
    app = run_module.HermesApp(
        make_settings(), AudioStreamManager(FakePyAudioModule()), detector, session,
        call_only=True, printer=lambda _: None,
    )
    assert app.run_once() is SessionEndReason.SILENCE
    assert session.speaker_id == 202
    assert detector.started and detector.stopped


def test_go_away_ends_session() -> None:
    from types import SimpleNamespace

    live = FakeLiveSession(turns=[[SimpleNamespace(go_away=object(), server_content=None)]])
    sess, _ = _make(FakeGenaiClient(live), vad_silence_timeout_sec=3.0)
    assert _run(sess) is SessionEndReason.SERVER_CLOSED


def test_half_duplex_mutes_mic_while_speaking(fast_timing: List[bytes]) -> None:
    sess, _ = _make(FakeGenaiClient(), half_duplex=True)

    async def go() -> None:
        sess._reset_state()
        sess._audio_queue.put_nowait(b"\x01\x00")  # Hermes com áudio pendente
        live = FakeLiveSession()
        task = asyncio.create_task(
            sess._record_and_send(live, FakeStream(pcm(3000)), None)  # type: ignore[arg-type]
        )
        await asyncio.sleep(0.05)
        assert fast_timing == []
        sess._drain_audio_queue()
        await asyncio.sleep(0.1)
        sess._stop.set()
        await task
        assert fast_timing, "microfone deveria voltar a enviar após a fala"

    asyncio.run(go())


def test_preroll_captures_audio_during_websocket_handshake(
    fast_timing: List[bytes],
) -> None:
    """Garante que a fala dita durante a conexão WebSocket não é perdida."""
    client = FakeGenaiClient()
    orig_aenter = client.aio.live.connect

    def slow_connect(**kwargs: Any) -> Any:
        cm = orig_aenter(**kwargs)
        orig_enter = cm.__aenter__

        async def delayed_enter() -> Any:
            await asyncio.sleep(0.04)  # Simula tempo de handshake WebSocket
            return await orig_enter()

        cm.__aenter__ = delayed_enter  # type: ignore[method-assign]
        return cm

    client.aio.live.connect = slow_connect  # type: ignore[assignment]
    pa = FakePyAudioModule(input_data=b"\x07\x00" * 1024)
    sess = GeminiLiveSession(
        make_settings(vad_silence_timeout_sec=0.15), AudioStreamManager(pa), client
    )
    assert _run(sess) is SessionEndReason.SILENCE
    assert len(fast_timing) >= 5
    assert fast_timing[0] == b"\x07\x00" * 1024


def test_barge_in_interrupts_assistant_playback_without_headphones(
    fast_timing: List[bytes],
) -> None:
    """Mesmo em half-duplex, falar alto na faixa de voz corta o playback do Hermes."""
    import numpy as np

    sess, _ = _make(FakeGenaiClient(), half_duplex=True, barge_in_enabled=True)
    t = np.arange(1024, dtype=np.float64) / 16000.0
    voice_pcm = (4500 * np.sin(2 * np.pi * 300 * t)).astype(np.int16).tobytes()

    async def go() -> None:
        sess._reset_state()
        sess._audio_queue.put_nowait(b"\x01\x00" * 100)
        sess._audio_queue.put_nowait(b"\x02\x00" * 100)
        live = FakeLiveSession()
        task = asyncio.create_task(
            sess._record_and_send(live, FakeStream(voice_pcm), None)  # type: ignore[arg-type]
        )
        await asyncio.sleep(0.06)
        sess._stop.set()
        await task
        # Fila de reprodução deve ter sido esvaziada pelo barge-in e a voz enviada
        assert sess._audio_queue.empty()
        assert len(fast_timing) >= 1

    asyncio.run(go())


# ------------------------------------------------------------- orquestrador
class FakeDetector:
    label = "Hermes"

    def __init__(self) -> None:
        self.calls = 0

    def wait_for_wake_word(self) -> None:
        self.calls += 1


def test_orchestrator_cycles_standby_active_standby(capsys: pytest.CaptureFixture) -> None:
    client = FakeGenaiClient()
    sess, pa = _make(client)
    det = FakeDetector()
    printed: List[str] = []
    app = run_module.HermesApp(
        make_settings(), AudioStreamManager(pa), det, sess, printer=printed.append  # type: ignore[arg-type]
    )
    assert app.state is run_module.AgentState.STANDBY
    reason = app.run_once()
    assert reason is SessionEndReason.SILENCE
    assert det.calls == 1
    assert app.state is run_module.AgentState.STANDBY
    assert printed[0].startswith("[Ren] Em espera")
    assert any("Sessão encerrada" in p for p in printed)


def test_orchestrator_no_network_in_standby() -> None:
    """Durante a espera da wake word nenhuma conexão é aberta."""
    client = FakeGenaiClient()
    sess, pa = _make(client)

    class CheckingDetector(FakeDetector):
        def wait_for_wake_word(self) -> None:
            assert not client.connected
            super().wait_for_wake_word()

    app = run_module.HermesApp(
        make_settings(), AudioStreamManager(pa), CheckingDetector(), sess, printer=lambda m: None  # type: ignore[arg-type]
    )
    app.run_once()
    assert client.connected


def test_call_rejoins_before_waiting_for_wake_word() -> None:
    client = FakeGenaiClient()
    sess, pa = _make(client)

    class Bot:
        is_in_call = False
        is_running = True
        joined = 0

        def join_call(self) -> str:
            self.joined += 1
            self.is_in_call = True
            return "conectado"

    bot = Bot()
    app = run_module.HermesApp(
        make_settings(), AudioStreamManager(pa), FakeDetector(), sess,
        printer=lambda _: None, call_only=True, discord_bot=bot,  # type: ignore[arg-type]
    )
    app.run_once = lambda: SessionEndReason.SHUTDOWN  # type: ignore[method-assign]
    app.run_forever()
    assert bot.joined == 1


def test_requested_leave_does_not_rejoin_call() -> None:
    class Bot:
        is_in_call = True
        joined = 0
        left = 0

        def join_call(self) -> str:
            self.joined += 1
            return "conectado"

        def leave_call(self) -> str:
            self.left += 1
            self.is_in_call = False
            return "desconectado"

    bot = Bot()
    sess, pa = _make(FakeGenaiClient())
    app = run_module.HermesApp(
        make_settings(), AudioStreamManager(pa), FakeDetector(), sess,
        printer=lambda _: None, call_only=True, discord_bot=bot,  # type: ignore[arg-type]
    )
    app.run_once = lambda: SessionEndReason.LEFT_CALL  # type: ignore[method-assign]
    app.run_forever()
    assert bot.left == 1
    assert bot.joined == 0


def test_build_config_gemini_38_and_safety_and_vad() -> None:
    from google.genai import types

    sess, _ = _make(
        FakeGenaiClient(),
        gemini_model="gemini-3.8-live-extended-thinking",
        thinking_level="LOW",
        safety_threshold="BLOCK_ONLY_HIGH",
    )
    cfg = _REAL_BUILD_CONFIG(sess)

    assert cfg.thinking_config is not None
    assert str(cfg.thinking_config.thinking_level).endswith("LOW")
    assert cfg.context_window_compression is not None
    assert cfg.context_window_compression.sliding_window is not None
    assert cfg.session_resumption is not None
    assert cfg.session_resumption.handle is None

    auto_vad = cfg.realtime_input_config.automatic_activity_detection
    assert auto_vad is not None
    assert auto_vad.start_of_speech_sensitivity == types.StartSensitivity.START_SENSITIVITY_LOW

    sess_no_think, _ = _make(FakeGenaiClient(), gemini_model="gemini-3.8-live")
    assert _REAL_BUILD_CONFIG(sess_no_think).thinking_config is None


def test_settings_validates_thinking_safety_and_vault(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.config import Settings

    vault = tmp_path / "ArgonVault"
    vault.mkdir()
    monkeypatch.setenv("GEMINI_API_KEY", "k-test")
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vault))
    monkeypatch.setenv("THINKING_LEVEL", "low")
    monkeypatch.setenv("SAFETY_THRESHOLD", "block_only_high")

    settings = Settings.from_env(load_dotenv_file=False)
    assert settings.obsidian_vault_path == vault
    assert settings.thinking_level == "LOW"
    assert settings.safety_threshold == "BLOCK_ONLY_HIGH"
    settings.validate()

    monkeypatch.setenv("THINKING_LEVEL", "INVALID")
    with pytest.raises(ValueError, match="THINKING_LEVEL"):
        Settings.from_env(load_dotenv_file=False).validate()

    monkeypatch.setenv("THINKING_LEVEL", "LOW")
    monkeypatch.setenv("SAFETY_THRESHOLD", "INVALID")
    with pytest.raises(ValueError, match="SAFETY_THRESHOLD"):
        Settings.from_env(load_dotenv_file=False).validate()


def test_transparent_reconnect_on_go_away_with_resumption_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace
    from src.gateway import GatewayStatus

    monkeypatch.setattr(GeminiLiveSession, "build_config", _REAL_BUILD_CONFIG)
    connect_handles: List[Any] = []
    gateway_states: List[GatewayStatus] = []

    class FakeGateway:
        def emit_state_changed(self, state: GatewayStatus) -> None:
            gateway_states.append(state)

        def __getattr__(self, _name: str) -> Any:
            return lambda *_args, **_kwargs: None

    live1 = FakeLiveSession(
        turns=[[
            SimpleNamespace(
                session_resumption_update=SimpleNamespace(new_handle="handle-abc-123"),
                go_away=None,
                server_content=None,
            ),
            SimpleNamespace(
                session_resumption_update=None,
                go_away=SimpleNamespace(time_left="5s"),
                server_content=None,
            ),
        ]]
    )
    live2 = FakeLiveSession(
        turns=[[
            SimpleNamespace(
                session_resumption_update=None,
                go_away=None,
                server_content=SimpleNamespace(
                    model_turn=None,
                    turn_complete=True,
                    interrupted=False,
                    input_transcription=None,
                    output_transcription=None,
                ),
            ),
        ]]
    )

    sessions = [live1, live2]

    class MultiConnectClient:
        def __init__(self) -> None:
            self.aio = SimpleNamespace(live=SimpleNamespace(connect=self._connect))

        def _connect(self, *, model: str, config: Any) -> Any:
            handle = config.session_resumption.handle if config.session_resumption else None
            connect_handles.append(handle)
            live = sessions.pop(0)

            class _Ctx:
                async def __aenter__(self) -> Any:
                    return live

                async def __aexit__(self, *args: Any) -> None:
                    return None

            return _Ctx()

    sess = GeminiLiveSession(
        make_settings(vad_silence_timeout_sec=0.1),
        AudioStreamManager(FakePyAudioModule()),
        MultiConnectClient(),  # type: ignore[arg-type]
        gateway=FakeGateway(),  # type: ignore[arg-type]
    )
    reason = _run(sess)

    assert reason is SessionEndReason.SILENCE
    assert connect_handles == [None, "handle-abc-123"]
    assert sess.resumption_handle == "handle-abc-123"
    # STANDBY só deve ser emitido uma única vez no encerramento final
    assert gateway_states.count(GatewayStatus.STANDBY) == 1
    assert gateway_states[0] is GatewayStatus.LISTENING


def test_transparent_reconnect_on_connection_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace
    from src.gateway import GatewayStatus

    monkeypatch.setattr(GeminiLiveSession, "build_config", _REAL_BUILD_CONFIG)

    class ConnectionClosedError(Exception):
        pass

    connect_handles: List[Any] = []
    gateway_states: List[GatewayStatus] = []

    class FakeGateway:
        def emit_state_changed(self, state: GatewayStatus) -> None:
            gateway_states.append(state)

        def __getattr__(self, _name: str) -> Any:
            return lambda *_args, **_kwargs: None

    class DroppingLiveSession(FakeLiveSession):
        async def _receive_iter(self) -> Any:
            yield SimpleNamespace(
                session_resumption_update=SimpleNamespace(new_handle="handle-drop-999"),
                go_away=None,
                server_content=None,
            )
            raise ConnectionClosedError("websocket 1011 internal error")

        def receive(self) -> Any:
            return self._receive_iter()

    live1 = DroppingLiveSession()
    live2 = FakeLiveSession(
        turns=[[
            SimpleNamespace(
                session_resumption_update=None,
                go_away=None,
                server_content=SimpleNamespace(
                    model_turn=None,
                    turn_complete=True,
                    interrupted=False,
                    input_transcription=None,
                    output_transcription=None,
                ),
            ),
        ]]
    )
    sessions = [live1, live2]

    class MultiConnectClient:
        def __init__(self) -> None:
            self.aio = SimpleNamespace(live=SimpleNamespace(connect=self._connect))

        def _connect(self, *, model: str, config: Any) -> Any:
            handle = config.session_resumption.handle if config.session_resumption else None
            connect_handles.append(handle)
            live = sessions.pop(0)

            class _Ctx:
                async def __aenter__(self) -> Any:
                    return live

                async def __aexit__(self, *args: Any) -> None:
                    return None

            return _Ctx()

    sess = GeminiLiveSession(
        make_settings(vad_silence_timeout_sec=0.1),
        AudioStreamManager(FakePyAudioModule()),
        MultiConnectClient(),  # type: ignore[arg-type]
        gateway=FakeGateway(),  # type: ignore[arg-type]
    )
    reason = _run(sess)

    assert reason is SessionEndReason.SILENCE
    assert connect_handles == [None, "handle-drop-999"]
    assert gateway_states.count(GatewayStatus.STANDBY) == 1
