from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest

from src.discord_bot.audio import (
    DISCORD_FRAME_BYTES,
    HermesDiscordAudioSource,
    HermesDiscordVoiceReceiver,
    convert_24k_mono_to_48k_stereo,
    convert_48k_stereo_to_16k_mono,
    ensure_opus_loaded,
    mix_pcm_16k,
)
from src.discord_bot.bot import DiscordVoiceBot
from src.gemini.knowledge import KnowledgeManager
from tests.conftest import make_settings


def test_ensure_opus_loaded() -> None:
    # A DLL oficial do discord.py está incluída no ambiente virtual Windows
    assert ensure_opus_loaded() is True


def test_convert_24k_mono_to_48k_stereo() -> None:
    assert convert_24k_mono_to_48k_stereo(b"") == b""
    assert convert_24k_mono_to_48k_stereo(b"\x01") == b""

    # 240 amostras em 24 kHz = 480 bytes (10 ms)
    raw_24k = (1500 * np.sin(np.linspace(0, 2 * np.pi, 240))).astype(np.int16).tobytes()
    out_48k = convert_24k_mono_to_48k_stereo(raw_24k)

    # A conversão 2x no tempo e 2x nos canais gera exatamente 4x bytes
    assert len(out_48k) == len(raw_24k) * 4

    samples_48k = np.frombuffer(out_48k, dtype=np.int16)
    samples_24k = np.frombuffer(raw_24k, dtype=np.int16)

    # Primeiro frame estéreo (Left e Right) deve ser igual à primeira amostra 24k
    assert samples_48k[0] == samples_24k[0]
    assert samples_48k[1] == samples_24k[0]
    # Segundo frame estéreo (repetição no tempo para 48k)
    assert samples_48k[2] == samples_24k[0]
    assert samples_48k[3] == samples_24k[0]


def test_convert_48k_stereo_to_16k_mono() -> None:
    assert convert_48k_stereo_to_16k_mono(b"") == b""
    assert convert_48k_stereo_to_16k_mono(b"\x00\x00") == b""

    # 960 amostras estéreo a 48 kHz (20 ms padrão do Discord) = 960 * 2 * 2 = 3840 bytes
    stereo_samples = np.full(1920, 3000, dtype=np.int16).tobytes()
    out_16k = convert_48k_stereo_to_16k_mono(stereo_samples)

    # 20 ms a 16 kHz mono = 320 amostras = 640 bytes
    assert len(out_16k) == 640
    samples_16k = np.frombuffer(out_16k, dtype=np.int16)
    assert samples_16k[0] == 3000


def test_mix_pcm_16k() -> None:
    assert mix_pcm_16k(b"", b"") == b""
    raw_a = np.array([1000, -500], dtype=np.int16).tobytes()
    raw_b = np.array([2000, -1000], dtype=np.int16).tobytes()

    mixed = mix_pcm_16k(raw_a, raw_b)
    samples = np.frombuffer(mixed, dtype=np.int16)
    assert samples[0] == 3000
    assert samples[1] == -1500

    # Clipping de saturação acima de 32767
    loud_a = np.array([30000], dtype=np.int16).tobytes()
    loud_b = np.array([10000], dtype=np.int16).tobytes()
    res = mix_pcm_16k(loud_a, loud_b)
    assert np.frombuffer(res, dtype=np.int16)[0] == 32767


def test_hermes_discord_audio_source() -> None:
    source = HermesDiscordAudioSource()
    assert source.is_opus() is False

    # Quando vazio, emite 3840 bytes de silêncio para manter o stream do Discord ativo sem estalos
    silence = source.read()
    assert len(silence) == DISCORD_FRAME_BYTES
    assert silence == b"\x00" * DISCORD_FRAME_BYTES

    # Alimenta 960 amostras em 24k (1920 bytes) -> converte para 3840 amostras 48k stereo (7680 bytes = 2 frames)
    raw_24k = (2000 * np.ones(960, dtype=np.int16)).tobytes()
    source.feed(raw_24k)

    frame1 = source.read()
    assert len(frame1) == DISCORD_FRAME_BYTES
    assert frame1 != b"\x00" * DISCORD_FRAME_BYTES

    frame2 = source.read()
    assert len(frame2) == DISCORD_FRAME_BYTES
    assert frame2 != b"\x00" * DISCORD_FRAME_BYTES

    # Terceiro frame volta a ser silêncio
    frame3 = source.read()
    assert frame3 == b"\x00" * DISCORD_FRAME_BYTES


def test_discord_output_discards_stale_audio_when_backlogged() -> None:
    source = HermesDiscordAudioSource(max_buffer_bytes=DISCORD_FRAME_BYTES * 2)
    source.feed((1000 * np.ones(960 * 3, dtype=np.int16)).tobytes())
    assert len(source._buffer) <= DISCORD_FRAME_BYTES * 2
    assert len(source.read()) == DISCORD_FRAME_BYTES


def test_discord_output_buffers_start_of_reply() -> None:
    source = HermesDiscordAudioSource(startup_frames=2)
    raw = (1000 * np.ones(960, dtype=np.int16)).tobytes()
    source.feed(raw)
    assert source.is_speaking_active
    assert source.read() == b"\x00" * DISCORD_FRAME_BYTES
    source.feed(raw)
    assert source.read() == b"\x00" * DISCORD_FRAME_BYTES
    assert source.read() != b"\x00" * DISCORD_FRAME_BYTES


def test_discord_output_keeps_bursty_sentence() -> None:
    source = HermesDiscordAudioSource()
    # Cinco segundos recebidos de uma vez não podem perder o começo da frase.
    source.feed((1000 * np.ones(24_000 * 5, dtype=np.int16)).tobytes())
    assert len(source._buffer) == DISCORD_FRAME_BYTES * 250


def test_discord_player_stays_active_between_chunks_until_turn_ends() -> None:
    source = HermesDiscordAudioSource()
    source.feed((1000 * np.ones(480, dtype=np.int16)).tobytes())
    source.read()
    for _ in range(30):
        source.read()
    assert source.is_speaking_active
    source.end_turn()
    assert not source.is_speaking_active


def test_hermes_discord_voice_receiver() -> None:
    receiver = HermesDiscordVoiceReceiver()
    assert receiver.wants_opus() is True
    assert receiver.has_audio() is False
    assert receiver.read_chunk(100) == b""

    # Usuário bot deve ser ignorado
    bot_user = MagicMock(bot=True)
    voice_data = MagicMock(opus=None, pcm=np.full(1920, 2000, dtype=np.int16).tobytes())
    receiver.write(bot_user, voice_data)
    assert receiver.has_audio() is False

    # Usuário humano deve ser processado e convertido para 16 kHz
    human_user = MagicMock(bot=False, display_name="Vinícius")
    receiver.write(human_user, voice_data)
    assert receiver.has_audio() is True
    assert receiver.last_speaker_name == "Vinícius"
    assert receiver.last_speech_time > 0.0

    chunk = receiver.read_chunk(640)
    assert len(chunk) == 640
    assert receiver.has_audio() is False


def test_hermes_discord_voice_receiver_dave_decryption_and_multi_ssrc() -> None:
    receiver = HermesDiscordVoiceReceiver()

    # Pacotes de silêncio Opus de 3 bytes (b"\xf8\xff\xfe") devem ser ignorados
    human_user = MagicMock(bot=False, id=123456, display_name="Vinícius")
    silence_pkt = MagicMock(opus=b"\xf8\xff\xfe", pcm=None, packet=MagicMock(ssrc=111))
    receiver.write(human_user, silence_pkt)
    assert receiver.has_audio() is False

    # Simula descriptografia DAVE E2EE ativa na VoiceConnectionState
    fake_dave = MagicMock()
    fake_dave.ready = True
    fake_dave.decrypt.side_effect = lambda uid, media_type, data: data[4:]  # remove prefixo "DAVE"

    fake_conn = MagicMock(dave_session=fake_dave, dave_protocol_version=1)
    fake_vc = MagicMock(_connection=fake_conn, guild=None)
    receiver.set_voice_client(fake_vc)

    decrypted = receiver._decrypt_dave_if_needed(fake_vc, 111, human_user, b"DAVERAW_OPUS")
    assert decrypted == b"RAW_OPUS"
    fake_dave.decrypt.assert_called_once()

    # Simula dois falantes simultâneos (SSRCs 101 e 202) enviando áudio ao mesmo tempo
    user_a = MagicMock(bot=False, id=101, display_name="Xanico")
    user_b = MagicMock(bot=False, id=202, display_name="Facimoto")
    data_a = MagicMock(
        opus=None,
        pcm=np.full(1920, 1000, dtype=np.int16).tobytes(),
        packet=MagicMock(ssrc=101),
    )
    data_b = MagicMock(
        opus=None,
        pcm=np.full(1920, 1500, dtype=np.int16).tobytes(),
        packet=MagicMock(ssrc=202),
    )
    receiver.write(user_a, data_a)
    receiver.write(user_b, data_b)

    mixed_chunk = receiver.read_chunk(640)
    assert len(mixed_chunk) == 640
    mixed_samples = np.frombuffer(mixed_chunk, dtype=np.int16)
    # 1000 + 1500 = 2500 (mixados simultaneamente sem desacelerar a voz!)
    assert mixed_samples[0] == 2500


def test_call_audio_keeps_speakers_separate_for_wake_and_gemini() -> None:
    receiver = HermesDiscordVoiceReceiver()
    for uid, value, name in ((101, 1000, "Ana"), (202, 2000, "Bruno")):
        user = MagicMock(bot=False, id=uid, display_name=name)
        packet = MagicMock(
            opus=None,
            pcm=np.full(1920, value, dtype=np.int16).tobytes(),
            packet=MagicMock(ssrc=uid),
        )
        receiver.write(user, packet)
    wake = receiver.read_wake_chunks(640)
    assert set(wake) == {101, 202}
    assert np.frombuffer(wake[101], dtype=np.int16)[0] == 1000
    assert np.frombuffer(wake[202], dtype=np.int16)[0] == 2000
    assert np.frombuffer(receiver.read_speaker_chunk(101, 640), dtype=np.int16)[0] == 1000
    assert np.frombuffer(receiver.read_speaker_chunk(202, 640), dtype=np.int16)[0] == 2000
    assert receiver.speaker_name(202) == "Bruno"
    receiver.reset_call_input()
    assert receiver.read_wake_chunks(640) == {}
    assert receiver.read_speaker_chunk(101, 640) == b""


def test_call_does_not_guess_author_for_unidentified_packet() -> None:
    receiver = HermesDiscordVoiceReceiver()
    user = MagicMock(bot=False, id=None, display_name="Desconhecido")
    packet = MagicMock(
        opus=None,
        pcm=np.full(1920, 1000, dtype=np.int16).tobytes(),
        packet=MagicMock(ssrc=999),
    )
    receiver.write(user, packet)
    assert receiver.read_wake_chunks(640) == {}
    assert receiver.read_speaker_chunk(999, 640) == b""


def test_discord_bot_unconfigured_token_graceful_messages(tmp_path: Path) -> None:
    settings = make_settings()
    bot = DiscordVoiceBot(settings)
    assert bot.is_running is False
    assert bot.is_in_call is False
    assert bot.has_incoming_audio() is False
    assert bot.get_incoming_chunk(100) == b""

    join_msg = bot.join_call("Geral")
    assert "DISCORD_BOT_TOKEN" in join_msg

    leave_msg = bot.leave_call()
    assert "não está conectado" in leave_msg

    status_msg = bot.get_status()
    assert "não configurado" in status_msg


def test_knowledge_manager_discord_tools_integration(tmp_path: Path) -> None:
    settings = make_settings()
    bot = DiscordVoiceBot(settings)
    km = KnowledgeManager(
        memory_file=tmp_path / "mem.json",
        knowledge_dir=tmp_path / "knowledge",
        discord_bot=bot,
    )

    names = [fn.name for fn in km.build_function_declarations()]
    assert "join_discord_call" in names
    assert "leave_discord_call" in names
    assert "get_discord_status" in names

    res_join = km.execute_tool("join_discord_call", {"channel_name": "Geral"})
    assert "result" in res_join
    assert "DISCORD_BOT_TOKEN" in res_join["result"]

    res_status = km.execute_tool("get_discord_status", {})
    assert "result" in res_status
    assert "DISCORD_BOT_TOKEN" in res_status["result"]

    res_leave = km.execute_tool("leave_discord_call", {})
    assert "result" in res_leave


def test_is_addressed_to_hermes() -> None:
    from src.gemini.prompts import is_addressed_to_hermes

    # Casos positivos (Hermes chamado de forma direta ou embutida)
    assert is_addressed_to_hermes("Hermes, que horas são?") is True
    assert is_addressed_to_hermes("qual é a previsão do tempo hermes?") is True
    assert is_addressed_to_hermes("Hermis me ajuda com isso") is True
    assert is_addressed_to_hermes("Ermes tá me ouvindo?") is True
    assert is_addressed_to_hermes("o que você acha, Hermes?") is True
    assert is_addressed_to_hermes("Hermes!") is True

    # Casos negativos (conversa entre outras pessoas na call sem chamar o Hermes)
    assert is_addressed_to_hermes("E aí galera, vamos jogar?") is False
    assert is_addressed_to_hermes("passa a call do meio") is False
    assert is_addressed_to_hermes("alguém viu o jogo ontem?") is False
    assert is_addressed_to_hermes("tô com fome vou almoçar") is False
    assert is_addressed_to_hermes("") is False


def test_discord_call_turn_gating_only_responds_when_hermes_addressed() -> None:
    from types import SimpleNamespace
    from src.audio.stream_manager import AudioStreamManager
    from src.gemini.session import GeminiLiveSession
    from tests.conftest import FakeGenaiClient, FakePyAudioModule

    settings = make_settings()
    bot = DiscordVoiceBot(settings)
    # Simula bot ativo em call
    bot._active_voice_channel_name = "Geral"
    bot._active_guild_name = "Servidor"

    session = GeminiLiveSession(
        settings,
        AudioStreamManager(FakePyAudioModule()),
        FakeGenaiClient(),
        discord_bot=bot,
    )
    session._turn_addressed_to_hermes = False

    # 1. Mensagem de outra pessoa na call sem citar "Hermes"
    msg_other = SimpleNamespace(
        go_away=None,
        server_content=SimpleNamespace(
            interrupted=False,
            grounding_metadata=None,
            input_transcription=SimpleNamespace(text="E aí galera, quem vai de suporte hoje?"),
            output_transcription=None,
            model_turn=SimpleNamespace(
                parts=[SimpleNamespace(inline_data=SimpleNamespace(data=b"\x01\x02" * 100))]
            ),
            turn_complete=True,
        ),
    )
    session._handle_message(msg_other)

    # Áudio NÃO deve ter sido enviado ao Discord (Hermes permaneceu mudo)
    assert bot._audio_source.is_speaking_active is False
    assert len(bot._audio_source._buffer) == 0

    # 2. Mensagem direcionada ao "Hermes"
    msg_hermes = SimpleNamespace(
        go_away=None,
        server_content=SimpleNamespace(
            interrupted=False,
            grounding_metadata=None,
            input_transcription=SimpleNamespace(text="Hermes, me diz a capital do Brasil"),
            output_transcription=SimpleNamespace(text="A capital é Brasília."),
            model_turn=SimpleNamespace(
                parts=[SimpleNamespace(inline_data=SimpleNamespace(data=b"\x05\x06" * 960))]
            ),
            turn_complete=True,
        ),
    )
    session._handle_message(msg_hermes)

    # Áudio DEVE ter sido enviado ao Discord com sucesso!
    assert bot._audio_source.is_speaking_active is True
    assert len(bot._audio_source._buffer) > 0


def test_call_wake_authorizes_first_question_without_name() -> None:
    from types import SimpleNamespace
    from src.audio.stream_manager import AudioStreamManager
    from src.gemini.session import GeminiLiveSession
    from tests.conftest import FakeGenaiClient, FakePyAudioModule

    bot = DiscordVoiceBot(make_settings())
    bot._active_voice_channel_name = "Geral"
    bot._active_guild_name = "Servidor"
    session = GeminiLiveSession(
        make_settings(), AudioStreamManager(FakePyAudioModule()),
        FakeGenaiClient(), discord_bot=bot, call_only=True,
    )
    assert session._turn_addressed_to_hermes

    def message(text: str, complete: bool) -> SimpleNamespace:
        return SimpleNamespace(
            go_away=None,
            server_content=SimpleNamespace(
                interrupted=False, grounding_metadata=None,
                input_transcription=SimpleNamespace(text=text),
                output_transcription=None,
                model_turn=SimpleNamespace(parts=[SimpleNamespace(
                    inline_data=SimpleNamespace(data=b"\x01\x00" * 480)
                )]),
                turn_complete=complete,
            ),
        )

    session._handle_message(message("Qual é a capital do Brasil?", True))
    assert bot._audio_source.is_speaking_active
    assert not session._turn_addressed_to_hermes


def test_call_publishes_requested_answer_and_real_sources_once() -> None:
    from types import SimpleNamespace
    from src.audio.stream_manager import AudioStreamManager
    from src.gemini.session import GeminiLiveSession
    from tests.conftest import FakeGenaiClient, FakePyAudioModule

    bot = DiscordVoiceBot(make_settings())
    bot._active_voice_channel_name = "Geral"
    bot._active_guild_name = "Servidor"
    sent: list[str] = []
    bot.publish_text = lambda content: sent.append(content) or True  # type: ignore[method-assign]
    session = GeminiLiveSession(
        make_settings(), AudioStreamManager(FakePyAudioModule()),
        FakeGenaiClient(), discord_bot=bot, call_only=True,
    )
    content = SimpleNamespace(
        interrupted=False,
        grounding_metadata=SimpleNamespace(
            web_search_queries=["capital Brasil"],
            grounding_chunks=[SimpleNamespace(web=SimpleNamespace(
                title="Fonte oficial", uri="https://example.org/fato"
            ))],
        ),
        input_transcription=SimpleNamespace(text="Pesquisa e manda no chat"),
        output_transcription=SimpleNamespace(text="A resposta é Brasília."),
        model_turn=None,
        turn_complete=True,
    )
    session._handle_message(SimpleNamespace(go_away=None, server_content=content))
    assert len(sent) == 1
    assert "A resposta é Brasília." in sent[0]
    assert "https://example.org/fato" in sent[0]
    session._handle_message(SimpleNamespace(go_away=None, server_content=SimpleNamespace(
        interrupted=False, grounding_metadata=None, input_transcription=None,
        output_transcription=None, model_turn=None, turn_complete=True,
    )))
    assert len(sent) == 1


def test_discord_text_channel_is_scoped_and_mentions_disabled() -> None:
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    settings = make_settings(discord_text_channel_id=1298820565037223990)
    bot = DiscordVoiceBot(settings)
    channel = SimpleNamespace(
        guild=SimpleNamespace(id=42, me=object()),
        permissions_for=lambda _: SimpleNamespace(view_channel=True, send_messages=True),
        send=AsyncMock(),
    )
    bot._active_guild_id = 42
    bot._client = SimpleNamespace(get_channel=lambda _: channel)
    with pytest.raises(StopIteration):
        bot._publish_text("x" * 2100).send(None)
    assert channel.send.await_count == 2
    assert all(
        len(call.args[0]) <= 1900 and call.kwargs["allowed_mentions"].everyone is False
        for call in channel.send.await_args_list
    )

    bot._active_guild_id = 99
    with pytest.raises(ValueError, match="não pertence"):
        bot._publish_text("texto").send(None)


def test_call_followup_remains_authorized_during_slow_search() -> None:
    from types import SimpleNamespace
    from src.audio.stream_manager import AudioStreamManager
    from src.gemini.session import GeminiLiveSession
    from tests.conftest import FakeGenaiClient, FakePyAudioModule

    settings = make_settings()
    bot = DiscordVoiceBot(settings)
    bot._active_voice_channel_name = "Geral"
    bot._active_guild_name = "Servidor"
    session = GeminiLiveSession(
        settings, AudioStreamManager(FakePyAudioModule()),
        FakeGenaiClient(), discord_bot=bot,
    )
    session._call_followup_until = float("inf")

    def message(*, text: str = "", audio: bytes = b"", complete: bool = False) -> SimpleNamespace:
        return SimpleNamespace(
            go_away=None,
            server_content=SimpleNamespace(
                interrupted=False, grounding_metadata=None,
                input_transcription=SimpleNamespace(text=text) if text else None,
                output_transcription=None,
                model_turn=SimpleNamespace(parts=[SimpleNamespace(
                    inline_data=SimpleNamespace(data=audio)
                )]) if audio else None,
                turn_complete=complete,
            ),
        )

    session._handle_message(message(text="Pesquisa isso para mim"))
    session._call_followup_until = 0.0  # a pesquisa demorou além da janela
    session._handle_message(message(audio=b"\x01\x02" * 480, complete=True))
    assert bot._audio_source.is_speaking_active
    assert session._call_followup_until > 0.0


def test_local_pc_speaker_muted_when_in_discord_call() -> None:
    import asyncio
    from src.audio.stream_manager import AudioStreamManager
    from src.gemini.session import GeminiLiveSession, _END_OF_TURN_SENTINEL
    from tests.conftest import FakeGenaiClient, FakePyAudioModule

    settings = make_settings()
    bot = DiscordVoiceBot(settings)
    bot._active_voice_channel_name = "Geral"
    bot._active_guild_name = "Servidor"

    pa = FakePyAudioModule()
    streams = AudioStreamManager(pa)
    out_stream = streams.open_output()

    session = GeminiLiveSession(
        settings,
        streams,
        FakeGenaiClient(),
        discord_bot=bot,
    )

    # Enfileira um chunk de fala e o sentinela de fim
    session._audio_queue.put_nowait(b"\x12\x34" * 480)
    session._audio_queue.put_nowait(_END_OF_TURN_SENTINEL)

    # Executa _play por 1 ciclo
    asyncio.run(session._play(out_stream, None))  # type: ignore[arg-type]

    # Como o bot está em call, o alto-falante local do PC deve ter recebido ZERO bytes!
    assert len(out_stream.written) == 0

    # Agora simula o bot fora da call
    bot._active_voice_channel_name = ""
    bot._active_guild_name = ""
    session._stop.clear()

    session._audio_queue.put_nowait(b"\x56\x78" * 480)
    session._audio_queue.put_nowait(_END_OF_TURN_SENTINEL)
    asyncio.run(session._play(out_stream, None))  # type: ignore[arg-type]

    # Fora da call, o áudio toca normalmente no alto-falante local do PC
    assert len(out_stream.written) == 1
    assert out_stream.written[0] == b"\x56\x78" * 480


def test_discord_audio_buffered_until_transcription_arrives() -> None:
    from types import SimpleNamespace
    from src.audio.stream_manager import AudioStreamManager
    from src.gemini.session import GeminiLiveSession
    from tests.conftest import FakeGenaiClient, FakePyAudioModule

    settings = make_settings()
    bot = DiscordVoiceBot(settings)
    bot._active_voice_channel_name = "Geral"
    bot._active_guild_name = "Servidor"

    session = GeminiLiveSession(
        settings,
        AudioStreamManager(FakePyAudioModule()),
        FakeGenaiClient(),
        discord_bot=bot,
    )
    session._turn_addressed_to_hermes = False

    # 1. Packet 1: O modelo responde sem transcrição de entrada (comportamento padrão do WebSocket Live)
    msg_audio_live = SimpleNamespace(
        go_away=None,
        server_content=SimpleNamespace(
            interrupted=False,
            grounding_metadata=None,
            input_transcription=None,
            output_transcription=None,
            model_turn=SimpleNamespace(
                parts=[SimpleNamespace(inline_data=SimpleNamespace(data=b"\x07\x08" * 480))]
            ),
            turn_complete=True,
        ),
    )
    session._handle_message(msg_audio_live)

    # A transcrição pode chegar depois do áudio; mantém o primeiro chunk audível.
    assert bot._audio_source.is_speaking_active is True
    assert len(bot._audio_source._buffer) > 0

    # 2. Packet 2: Conversa paralela confirmada entre outros membros (sem citar Hermes)
    bot._audio_source.clear()
    msg_side_chatter = SimpleNamespace(
        go_away=None,
        server_content=SimpleNamespace(
            interrupted=False,
            grounding_metadata=None,
            input_transcription=SimpleNamespace(text="E aí galera, quem vai de suporte hoje?"),
            output_transcription=None,
            model_turn=SimpleNamespace(
                parts=[SimpleNamespace(inline_data=SimpleNamespace(data=b"\x09\x10" * 480))]
            ),
            turn_complete=True,
        ),
    )
    session._handle_message(msg_side_chatter)

    # Como o texto confirma que a fala não foi para o Hermes, o áudio é descartado!
    assert len(bot._audio_source._buffer) == 0


def test_discord_tool_call_never_dropped_in_call() -> None:
    import asyncio
    from types import SimpleNamespace
    from src.audio.stream_manager import AudioStreamManager
    from src.gemini.session import GeminiLiveSession
    from tests.conftest import FakeGenaiClient, FakePyAudioModule

    settings = make_settings()
    bot = DiscordVoiceBot(settings)
    bot._active_voice_channel_name = "Geral"
    bot._active_guild_name = "Servidor"

    session = GeminiLiveSession(
        settings,
        AudioStreamManager(FakePyAudioModule()),
        FakeGenaiClient(),
        discord_bot=bot,
    )
    session._turn_addressed_to_hermes = False

    tool_called_back = []

    async def _mock_send_tool(function_responses: Any) -> None:
        tool_called_back.extend(function_responses)

    fake_live = SimpleNamespace(
        send_tool_response=_mock_send_tool
    )

    # Simula tool call do Gemini (ex.: get_pc_status) no meio da call antes da transcrição chegar
    tool_call_obj = SimpleNamespace(
        function_calls=[SimpleNamespace(name="get_pc_status", args={}, id="call_123")]
    )

    async def _run_tool() -> None:
        await session._handle_tool_call(fake_live, tool_call_obj)

    asyncio.run(_run_tool())

    # A tool DEVE ter sido executada e respondida ao Gemini para não travar a IA
    assert len(tool_called_back) == 1
    assert tool_called_back[0].name == "get_pc_status"


def test_vad_monitor_does_not_timeout_in_discord_call() -> None:
    import asyncio
    from src.audio.stream_manager import AudioStreamManager
    from src.gemini.session import GeminiLiveSession
    from tests.conftest import FakeGenaiClient, FakePyAudioModule

    settings = make_settings(vad_silence_timeout_sec=0.05)
    bot = DiscordVoiceBot(settings)
    bot._active_voice_channel_name = "Geral"
    bot._active_guild_name = "Servidor"

    session = GeminiLiveSession(
        settings,
        AudioStreamManager(FakePyAudioModule()),
        FakeGenaiClient(),
        discord_bot=bot,
    )

    async def _test() -> None:
        task = asyncio.create_task(session._vad_monitor(None))
        await asyncio.sleep(0.15)
        # Em call, o VAD monitor NÃO deve disparar timeout nem encerrar a sessão
        assert not session._stop.is_set()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(_test())

