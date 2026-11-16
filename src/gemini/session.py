"""Sessão assíncrona com a Gemini Multimodal Live API.

Tarefas concorrentes (coordenadas por um `asyncio.Event` de parada):
- Pré-conexão: captura o microfone em buffer enquanto o WebSocket conecta,
  evitando perder o início da frase logo após a wake word.
- t_rec  (Task A): microfone 16 kHz -> AEC/barge-in -> envio de blobs PCM.
- t_recv (Task B1): recepção do WebSocket (+ Grounding / Tool Calls) -> fila de áudio.
- t_play (Task B2): fila de áudio -> alto-falante 24 kHz (+ referência pro AEC).
- t_vad  (Task C): encerra após N segundos sem voz de ninguém.
"""

from __future__ import annotations

import asyncio
import enum
import inspect
import logging
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Deque, List, Optional, Set

from src.audio.aec import AcousticEchoCanceller, enhance_voice_pcm
from src.audio.stream_manager import (
    AudioStreamManager,
    close_stream,
    read_chunk,
    write_chunk,
)
from src.audio.vad import SilenceMonitor
from src.config import CHUNK_SIZE, INPUT_MIME_TYPE, INPUT_RATE, OUTPUT_RATE, Settings
from src.core.router import EdgeRouter
from src.discord_bot.bot import DiscordVoiceBot
from src.gateway.ws_server import GatewayServer, GatewayStatus
from src.gemini.knowledge import KnowledgeManager
from src.gemini.prompts import (
    build_system_instruction,
    is_addressed_to_hermes,
    is_chat_request,
    is_farewell,
    is_leave_call_command,
    is_shutdown_command,
    load_friend_profiles,
)
from src.memory.compactor import ContextCompactor
from src.vision.screen import (
    ScreenCaptureWorker,
    capture_screen_jpeg,
    get_active_window_info,
    is_screen_request,
    send_jpeg_frame_to_session,
)

logger = logging.getLogger(__name__)

# Janela após o último chunk tocado em que ainda consideramos o Hermes falando
# (cobre a latência do buffer de saída do PortAudio + reverberação do quarto).
_SPEAKING_TAIL_SEC = 0.55
_POLL_SEC = 0.1
_SHUTDOWN_GRACE_SEC = 1.0
_CALL_FOLLOWUP_SEC = 45.0
# Máximo de chunks guardados durante o handshake (~9.6s a 16kHz / 1024 frames)
_MAX_PREROLL_CHUNKS = 150
_MAX_RECONNECT_ATTEMPTS = 5

_END_OF_TURN_SENTINEL: Optional[bytes] = None


def _is_connection_closed_error(exc: BaseException) -> bool:
    """True para quedas de WebSocket/rede passíveis de retomada transparente."""
    return isinstance(exc, ConnectionError) or "ConnectionClosed" in type(exc).__name__


class SessionEndReason(enum.Enum):
    SILENCE = "silêncio"
    FAREWELL = "despedida"
    SHUTDOWN = "encerramento total"
    LEFT_CALL = "saída da call"
    SERVER_CLOSED = "servidor encerrou"
    ERROR = "erro"


def _current_task_is_cancelling() -> bool:
    """True se a task atual recebeu um cancel() externo (ex.: Ctrl+C)."""
    task = asyncio.current_task()
    cancelling = getattr(task, "cancelling", None)  # Python 3.11+
    return bool(cancelling and cancelling())


class GeminiLiveSession:
    def __init__(
        self,
        settings: Settings,
        stream_manager: AudioStreamManager,
        client: Optional[Any] = None,
        knowledge_manager: Optional[KnowledgeManager] = None,
        gateway: Optional[GatewayServer] = None,
        edge_router: Optional[EdgeRouter] = None,
        vision_worker: Optional[ScreenCaptureWorker] = None,
        discord_bot: Optional[DiscordVoiceBot] = None,
        call_only: bool = False,
    ) -> None:
        self._settings = settings
        self._streams = stream_manager
        if client is None:
            from google import genai

            client = genai.Client(
                api_key=settings.gemini_api_key,
                http_options={"api_version": "v1beta"},
            )
        self._client = client
        self._gateway = gateway or GatewayServer(enabled=False)
        self._discord_bot = discord_bot
        self._friend_profiles = load_friend_profiles()
        self._call_only = call_only
        self._call_speaker_id: Optional[int] = None
        self._call_handoff_checker: Optional[Callable[[], bool]] = None
        self._knowledge = knowledge_manager or KnowledgeManager(
            gateway=self._gateway,
            discord_bot=self._discord_bot,
            obsidian_vault_path=settings.obsidian_vault_path,
            genai_client=self._client,
        )
        if self._knowledge.genai_client is None:
            self._knowledge.genai_client = self._client
        if self._knowledge.obsidian_vault_path is None and settings.obsidian_vault_path is not None:
            self._knowledge.obsidian_vault_path = settings.obsidian_vault_path
        if self._knowledge.gateway is None:
            self._knowledge.gateway = self._gateway
        if self._knowledge.discord_bot is None and self._discord_bot is not None:
            self._knowledge.discord_bot = self._discord_bot
        self._compactor = ContextCompactor()
        self._active_live_session: Optional[Any] = None
        self._edge_router = edge_router or EdgeRouter(gateway=self._gateway)
        self._vision_worker = vision_worker or ScreenCaptureWorker(
            interval_sec=1.0, max_width=1280, max_height=720, quality=60
        )
        self._aec = AcousticEchoCanceller(
            frame_size=CHUNK_SIZE,
            barge_in_rms_threshold=max(settings.vad_silence_threshold_rms * 1.4, 650.0),
        )
        self._pending_reminders: Deque[str] = deque()
        self._wake_callback: Optional[Callable[[], None]] = None
        self._knowledge.timers.set_voice_reminder_callback(self.trigger_reminder)
        self.resumption_handle: Optional[str] = None
        self._reconnect_requested: bool = False
        self._reset_state()

    def set_wake_callback(self, callback: Optional[Callable[[], None]]) -> None:
        """Registra callback (ex.: `detector.trigger_wakeup`) para acordar o Hermes do Standby quando um timer zerar."""
        self._wake_callback = callback

    def set_call_speaker(self, speaker_id: Optional[int]) -> None:
        """Fixa o membro que chamou Ren para a próxima sessão da call."""
        self._call_speaker_id = speaker_id

    def set_call_handoff_checker(self, checker: Optional[Callable[[], bool]]) -> None:
        self._call_handoff_checker = checker

    def trigger_reminder(self, label: str) -> None:
        """Enfileira um lembrete disparado por timer para o Hermes falar em voz alta imediatamente."""
        self._pending_reminders.append(label)
        self._gateway.emit_ui_widget(
            "timer_fired",
            {"label": label, "message": f"Lembrete disparado: {label}"},
        )
        if self._wake_callback is not None:
            self._wake_callback()

    # ------------------------------------------------------------------ setup
    def _reset_state(self) -> None:
        self._stop = asyncio.Event()
        self._audio_queue: "asyncio.Queue[Optional[bytes]]" = asyncio.Queue()
        self._preroll_buffer: Deque[bytes] = deque(maxlen=_MAX_PREROLL_CHUNKS)
        self._monitor = SilenceMonitor(
            threshold_rms=self._settings.vad_silence_threshold_rms,
            timeout_sec=self._settings.vad_silence_timeout_sec,
            silero_model_path=self._settings.silero_model_path,
        )
        self._aec.reset()
        self._last_playback = 0.0
        self._call_followup_until = 0.0
        self._call_idle_deadline = time.monotonic() + 30.0
        self._call_completed_reply = False
        self._farewell_requested = False
        self._shutdown_requested = False
        self._leave_call_requested = False
        in_call = self._discord_bot is not None and getattr(
            self._discord_bot, "is_in_call", False
        )
        # No processo da call, a wake word já foi validada localmente e pode
        # ter sido consumida antes de o Gemini receber a pergunta inicial.
        self._turn_addressed_to_hermes: bool = self._call_only or not in_call
        self._pending_turn_audio: Deque[bytes] = deque(maxlen=150)
        self._user_turn_text = ""
        self._assistant_turn_text = ""
        self._turn_had_audio = False
        self._turn_speaker_name = ""
        self._announced_speaker_id: Optional[int] = None
        self._announced_speaker_name = ""
        self._last_speaker_notify_ts: float = 0.0
        self._screen_sent_in_turn = False
        self._logged_queries: Set[str] = set()
        self._turn_sources: List[tuple[str, str]] = []
        self._chat_requested = False
        self._edge_router.reset_turn()
        self.resumption_handle = None
        self._reconnect_requested = False
        self._end_reason: SessionEndReason = SessionEndReason.ERROR

    def build_config(self) -> Any:
        from google.genai import types

        discord_ctx = ""
        if self._discord_bot is not None and getattr(self._discord_bot, "is_in_call", False):
            ch = getattr(self._discord_bot, "active_channel", "")
            g = getattr(self._discord_bot, "active_guild", "")
            discord_ctx = (
                f"Você está atualmente conectado no canal de voz '{ch}' do servidor '{g}' no Discord. "
                f"As pessoas na call são interlocutores diferentes. Não suponha que quem fala é Vinícius. "
                f"IMPORTANTE: responda a qualquer participante humano da call que chamar você por 'Ren', não apenas ao dono do bot. "
                + (
                    "A palavra 'Ren' que abriu esta sessão já foi detectada localmente; responda à primeira pergunta mesmo se a transcrição não contiver o nome. "
                    if self._call_speaker_id is not None
                    else ""
                )
                + f"O reconhecimento de voz pode transcrever esse nome curto como 'rei', 'rein', 'rain', 'pain', 'hey', 'ei', 'hein', 'hem', 'rende' ou 'renda'; trate esses termos como Ren quando usados como chamado. "
                f"Depois de responder a um chamado, aceite perguntas de acompanhamento por cerca de quarenta e cinco segundos sem exigir o nome novamente. "
                f"Conversa entre participantes sem chamado não exige resposta; nunca verbalize essa regra."
            )
            if self._call_speaker_id is not None:
                speaker_name = self._discord_bot.speaker_name(self._call_speaker_id)
                is_creator = self._call_speaker_id == self._settings.discord_user_id
                discord_ctx += (
                    f" A pessoa que chamou Ren nesta sessão é {speaker_name or 'um participante'} "
                    f"(ID Discord {self._call_speaker_id}). Dirija as respostas a essa pessoa. "
                    + (
                        "Este ID é o do Vinícius (seu criador e dono do bot). Reconheça-o como Vinícius."
                        if is_creator
                        else "Este ID não é o do Vinícius. O perfil, a localização e as memórias pessoais do criador não descrevem esta pessoa; não os atribua a ela."
                    )
                )
            known_friends = {profile["nome"]: profile["fatos"] for profile in self._friend_profiles.values()}
            if known_friends:
                discord_ctx += " Amigos cadastrados (use apenas quando a pessoa se identificar ou for mencionada): "
                discord_ctx += "; ".join(f"{name}: {facts}" for name, facts in known_friends.items())

        dynamic_prompt = build_system_instruction(
            memories=self._knowledge.list_memories(),
            knowledge_files=self._knowledge.list_knowledge_files(),
            conversation_context=self._knowledge.get_conversation_context(),
            discord_context=discord_ctx,
            in_discord_call=bool(discord_ctx),
        )

        is_gemini_38 = self._settings.gemini_model.startswith("gemini-3.8-live")
        live_tools = []
        if not is_gemini_38:
            live_tools.append(types.Tool(google_search=types.GoogleSearch()))
        live_tools.append(
            types.Tool(function_declarations=self._knowledge.build_function_declarations())
        )

        supports_thinking = "thinking" in self._settings.gemini_model.lower()
        return types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            temperature=0.4,
            system_instruction=types.Content(
                parts=[types.Part.from_text(text=dynamic_prompt)]
            ),
            tools=live_tools,
            thinking_config=(
                types.ThinkingConfig(
                    thinking_level=self._settings.thinking_level.upper()
                )
                if supports_thinking
                else None
            ),
            context_window_compression=types.ContextWindowCompressionConfig(
                sliding_window=types.SlidingWindow()
            ),
            session_resumption=types.SessionResumptionConfig(
                handle=self.resumption_handle
            ),
            speech_config=types.SpeechConfig(
                language_code=None if is_gemini_38 else "pt-BR",
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=self._settings.gemini_voice
                    )
                ),
            ),
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(
                    disabled=False,
                    start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_LOW,
                    end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_LOW,
                    prefix_padding_ms=200,
                    silence_duration_ms=280,
                ),
                turn_coverage=types.TurnCoverage.TURN_INCLUDES_AUDIO_ACTIVITY_AND_ALL_VIDEO,
            ),
            # Transcrições de entrada e saída: usadas para detectar despedidas
            # e manter o histórico reduzido de conversas entre sessões.
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
        )

    # ---------------------------------------------------------------- estado
    def assistant_is_speaking(self) -> bool:
        return (
            not self._audio_queue.empty()
            or time.monotonic() - self._last_playback < _SPEAKING_TAIL_SEC
            or (
                self._discord_bot is not None
                and getattr(self._discord_bot, "is_speaking", False)
            )
        )

    def _interrupt_playback(self) -> None:
        """Corta o áudio atual do Hermes imediatamente (barge-in ou interrupção)."""
        self._drain_audio_queue()
        self._pending_turn_audio.clear()
        self._last_playback = 0.0
        self._aec.reset()
        if self._discord_bot is not None:
            self._discord_bot._audio_source.clear()

    def _finish(self, reason: SessionEndReason) -> None:
        if not self._stop.is_set():
            self._end_reason = reason
            self._stop.set()

    def _request_reconnect(self) -> None:
        """Sinaliza reconexão transparente sem voltar para STANDBY."""
        if not self._stop.is_set():
            self._reconnect_requested = True
            self._stop.set()

    # ----------------------------------------------------------------- tasks
    async def _capture_preroll(
        self,
        in_stream: Any,
        executor: Optional[ThreadPoolExecutor],
        preroll_stop: asyncio.Event,
    ) -> None:
        """Grava o microfone (ou áudio da call do Discord) em memória enquanto o WebSocket conecta."""
        while not preroll_stop.is_set() and not self._stop.is_set():
            data = await self._read_input(in_stream, executor)
            if not self._call_only and self._discord_bot is not None and getattr(self._discord_bot, "is_in_call", False):
                discord_chunk = self._discord_bot.get_incoming_chunk(len(data))
                data = discord_chunk[: len(data)].ljust(len(data), b"\x00")
            self._preroll_buffer.append(data)

    async def _read_input(self, in_stream: Any, executor: Optional[ThreadPoolExecutor]) -> bytes:
        if self._call_only:
            await asyncio.sleep(CHUNK_SIZE / INPUT_RATE)
            assert self._discord_bot is not None
            if self._call_speaker_id is None:
                return b"\x00" * (CHUNK_SIZE * 2)
            return self._discord_bot.get_speaker_chunk(
                self._call_speaker_id, CHUNK_SIZE * 2
            ).ljust(CHUNK_SIZE * 2, b"\x00")
        return await read_chunk(in_stream, CHUNK_SIZE, executor)

    async def _send_audio(self, session: Any, data: bytes) -> None:
        from google.genai import types

        blob = types.Blob(data=data, mime_type=INPUT_MIME_TYPE)
        send_realtime = getattr(session, "send_realtime_input", None)
        if send_realtime is not None:
            await send_realtime(audio=blob)
        else:  # SDKs antigos
            await session.send(input=blob)

    async def _notify_call_speaker(
        self, session: Any, speaker_id: Optional[int], speaker_name: str
    ) -> None:
        """Informa silenciosamente ao Gemini quem está falando agora na call do Discord."""
        send_client = getattr(session, "send_client_content", None)
        if send_client is None or speaker_id is None:
            return
        now = time.monotonic()
        if self._last_speaker_notify_ts > 0.0 and now - self._last_speaker_notify_ts < 2.0:
            return
        self._last_speaker_notify_ts = now
        from google.genai import types

        is_creator = (
            self._settings.discord_user_id is not None
            and speaker_id == self._settings.discord_user_id
        )
        if is_creator:
            cue = (
                f"[Falante atual na call: {speaker_name or 'Vinícius'} (ID {speaker_id}, "
                f"seu criador Vinícius). Trate-o com o jeito mais fofo, amável e carinhoso.]"
            )
        else:
            cue = (
                f"[Falante atual na call: {speaker_name or 'Participante'} (ID {speaker_id}, "
                f"NÃO é o Vinícius). Mantenha o tom ácido, sarcástico e com palavrões pontuais.]"
            )
        res = send_client(
            turns=types.Content(
                role="user",
                parts=[types.Part.from_text(text=cue)],
            ),
            turn_complete=False,
        )
        if inspect.isawaitable(res):
            await res

    async def _flush_preroll(self, session: Any) -> None:
        """Envia o áudio capturado durante a conexão antes dos chunks ao vivo."""
        while self._preroll_buffer and not self._stop.is_set():
            chunk = enhance_voice_pcm(self._preroll_buffer.popleft())
            self._monitor.feed(chunk)
            await self._send_audio(session, chunk)

    async def _record_and_send(
        self, session: Any, in_stream: Any, executor: Optional[ThreadPoolExecutor]
    ) -> None:
        await self._flush_preroll(session)
        while not self._stop.is_set():
            data = await self._read_input(in_stream, executor)
            speaking = self.assistant_is_speaking()
            in_discord_call = self._discord_bot is not None and getattr(
                self._discord_bot, "is_in_call", False
            )
            if in_discord_call and self._discord_bot is not None and not self._call_only:
                # Drena continuamente o buffer do Discord (mesmo enquanto o Hermes fala,
                # para nunca acumular áudio velho).
                discord_chunk = self._discord_bot.get_incoming_chunk(len(data))
                # Na call, o microfone local não faz parte da conversa. Misturá-lo
                # introduz eco e ruído e pode ocultar a pergunta dos participantes.
                data = discord_chunk[: len(data)].ljust(len(data), b"\x00")

            if in_discord_call and self._discord_bot is not None and any(data):
                if self._call_only and self._call_speaker_id is not None:
                    speaker_id = self._call_speaker_id
                    speaker_name = self._discord_bot.speaker_name(speaker_id)
                else:
                    speaker_id = self._discord_bot.last_speaker_id
                    speaker_name = self._discord_bot.last_speaker_name
                self._call_idle_deadline = time.monotonic() + 30.0
                if speaker_name and (
                    speaker_id != self._announced_speaker_id
                    or speaker_name != self._announced_speaker_name
                ):
                    self._announced_speaker_id = speaker_id
                    self._announced_speaker_name = speaker_name
                    self._turn_speaker_name = speaker_name

            if in_discord_call:
                # O receptor do Discord já exclui a voz do bot. O AEC local
                # adiciona trabalho por chunk sem ter referência de eco aqui.
                cleaned, barge_in = enhance_voice_pcm(data), False
            else:
                cleaned, barge_in = self._aec.process(data, speaking)

            if speaking and barge_in and self._settings.barge_in_enabled and not in_discord_call:
                logger.info("Barge-in detectado: interrompendo fala do Hermes.")
                self._interrupt_playback()
                self._gateway.emit_state_changed(GatewayStatus.LISTENING)
                self._gateway.emit_audio_chunk_telemetry(cleaned, sample_rate=INPUT_RATE)
                self._monitor.feed(cleaned)
                await self._send_audio(session, cleaned)
                continue

            if speaking and (self._settings.half_duplex or in_discord_call):
                # Evita que o Hermes "ouça" o próprio eco residual pelo alto-falante.
                # Quando está em call no Discord, envia quadros de silêncio para manter o
                # relógio do VAD do servidor Gemini sincronizado sem interrupção.
                if in_discord_call:
                    try:
                        await self._send_audio(session, b"\x00" * len(data))
                    except Exception as exc:  # noqa: BLE001
                        if _is_connection_closed_error(exc) and self.resumption_handle:
                            self._request_reconnect()
                            return
                        raise
                continue

            if not speaking:
                self._gateway.emit_audio_chunk_telemetry(cleaned, sample_rate=INPUT_RATE)
            self._monitor.feed(cleaned)
            try:
                await self._send_audio(session, cleaned)
            except Exception as exc:  # noqa: BLE001
                if _is_connection_closed_error(exc) and self.resumption_handle:
                    self._request_reconnect()
                    return
                raise

    async def _receive(self, session: Any) -> None:
        try:
            while not self._stop.is_set():
                # `receive()` termina a cada turn_complete; por isso o loop externo.
                received_any = False
                async for message in session.receive():
                    received_any = True
                    self._monitor.mark_activity()
                    self._handle_message(message)
                    if (
                        not self._screen_sent_in_turn
                        and self._user_turn_text
                        and is_screen_request(self._user_turn_text)
                    ):
                        await self._send_screen_snapshot(session)
                    tool_call = getattr(message, "tool_call", None)
                    if tool_call is not None:
                        asyncio.create_task(
                            self._handle_tool_call(session, tool_call),
                            name="t_tool",
                        )
                    if self._stop.is_set():
                        return
                if not received_any:
                    # Iterador vazio = conexão encerrada pelo servidor.
                    if self.resumption_handle:
                        self._request_reconnect()
                    else:
                        self._finish(SessionEndReason.SERVER_CLOSED)
                    return
        except Exception as exc:  # noqa: BLE001
            if _is_connection_closed_error(exc) and self.resumption_handle:
                logger.warning(
                    "Conexão WebSocket fechada (%s); acionando reconexão transparente.",
                    exc,
                )
                self._request_reconnect()
                return
            raise

    async def _send_screen_snapshot(self, session: Any) -> bool:
        """Captura a tela em JPEG fora do event loop e envia para a sessão Live."""
        try:
            self._gateway.emit_state_changed(GatewayStatus.PROCESSING)
            _, win_title, _, _ = get_active_window_info()
            win_info = f" (Janela ativa: '{win_title}')" if win_title else ""
            print(
                f"[Ren] Capturando e analisando a sua tela{win_info}...",
                flush=True,
            )
            loop = asyncio.get_running_loop()
            jpeg_bytes = await loop.run_in_executor(None, capture_screen_jpeg)
            await send_jpeg_frame_to_session(session, jpeg_bytes)
            self._screen_sent_in_turn = True
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Falha ao capturar/enviar tela: %s", exc)
            return False

    async def _handle_tool_call(self, session: Any, tool_call: Any) -> None:
        """Executa ferramentas locais (memória / documentos / visão da tela) e devolve à Gemini."""
        from google.genai import types

        function_calls = getattr(tool_call, "function_calls", None) or []
        if not function_calls:
            return

        self._gateway.emit_state_changed(GatewayStatus.PROCESSING)
        responses = []
        for fc in function_calls:
            name = getattr(fc, "name", "")
            args = dict(getattr(fc, "args", None) or {})
            if name == "look_at_screen":
                await self._send_screen_snapshot(session)
                await asyncio.sleep(0.2)
            elif name == "close_assistant":
                mode = str(args.get("mode", "standby")).lower()
                self._farewell_requested = True
                if mode == "shutdown":
                    self._shutdown_requested = True
                    logger.info("Comando de desligamento total acionado via tool.")
                else:
                    logger.info("Comando de voltar ao standby acionado via tool.")
            elif name == "leave_discord_call" and self._call_only:
                # A desconexão imediata corta a despedida e o orquestrador
                # a interpreta como queda, reconectando na mesma call.
                self._leave_call_requested = True
                self._farewell_requested = True
                responses.append(
                    types.FunctionResponse(
                        id=getattr(fc, "id", None),
                        name=name,
                        response={"result": "Saída da call agendada após a despedida."},
                    )
                )
                continue
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, self._knowledge.execute_tool, name, args
            )
            self._monitor.mark_activity()
            responses.append(
                types.FunctionResponse(
                    id=getattr(fc, "id", None),
                    name=name,
                    response=result,
                )
            )
        send_tool = getattr(session, "send_tool_response", None)
        if send_tool is not None and responses:
            res = send_tool(function_responses=responses)
            if inspect.isawaitable(res):
                await res

    def _maybe_schedule_compaction(self) -> None:
        if self._active_live_session is None:
            return
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(
                self._compactor.maybe_compact_and_inject(
                    store=self._knowledge.sqlite_store,
                    genai_client=self._client,
                    live_session=self._active_live_session,
                ),
                name="t_compact",
            )
        except RuntimeError:
            pass

    def _handle_message(self, message: Any) -> None:
        content = getattr(message, "server_content", None)

        resumption_update = getattr(message, "session_resumption_update", None)
        if resumption_update is None and content is not None:
            resumption_update = getattr(content, "session_resumption_update", None)
        if resumption_update is not None:
            new_handle = getattr(resumption_update, "new_handle", None)
            if isinstance(new_handle, str) and new_handle:
                self.resumption_handle = new_handle

        go_away = getattr(message, "go_away", None)
        if go_away is None and content is not None:
            go_away = getattr(content, "go_away", None)
        if go_away is not None:
            if self.resumption_handle:
                logger.info("Servidor solicitou go_away; reconectando com resumption_handle.")
                self._request_reconnect()
            else:
                logger.info("Servidor solicitou encerramento da sessão.")
                self._finish(SessionEndReason.SERVER_CLOSED)
            return

        if content is None:
            return

        followup_active = time.monotonic() < self._call_followup_until

        if getattr(content, "interrupted", False):
            # O servidor já encerrou o turno anterior. Manter seu áudio na fila
            # mistura a resposta antiga com a próxima, inclusive na call.
            self._interrupt_playback()
            self._gateway.emit_state_changed(GatewayStatus.LISTENING)

        grounding = getattr(content, "grounding_metadata", None)
        if grounding is not None:
            for chunk in getattr(grounding, "grounding_chunks", None) or []:
                web = getattr(chunk, "web", None)
                uri = str(getattr(web, "uri", "") or "")
                title = str(getattr(web, "title", "") or "Fonte")
                if uri.startswith("https://") and uri not in {url for _, url in self._turn_sources}:
                    self._turn_sources.append((title[:100], uri))
            queries = getattr(grounding, "web_search_queries", None) or []
            new_queries = [q for q in queries if q and q not in self._logged_queries]
            if new_queries:
                self._logged_queries.update(new_queries)
                joined = ", ".join(f"'{q}'" for q in new_queries)
                print(f"[Ren] Pesquisando no Google: {joined}", flush=True)
                self._gateway.emit_state_changed(GatewayStatus.PROCESSING)
                self._gateway.emit_ui_widget(
                    "google_search", {"queries": list(new_queries)}
                )

        transcription = getattr(content, "input_transcription", None)
        text = getattr(transcription, "text", None) if transcription else None
        if text:
            self._user_turn_text += text
            if is_chat_request(self._user_turn_text):
                self._chat_requested = True
            if followup_active or is_addressed_to_hermes(self._user_turn_text):
                self._turn_addressed_to_hermes = True
            self._edge_router.try_dispatch(self._user_turn_text)
            if not self._shutdown_requested and is_shutdown_command(
                self._user_turn_text
            ):
                logger.info(
                    "Desligamento total detectado: %r", self._user_turn_text.strip()
                )
                self._shutdown_requested = True
                self._farewell_requested = True
            elif (
                self._call_only
                and self._turn_addressed_to_hermes
                and is_leave_call_command(self._user_turn_text)
            ):
                self._leave_call_requested = True
                self._farewell_requested = True
            elif not self._farewell_requested and is_farewell(self._user_turn_text):
                in_call = self._discord_bot is not None and getattr(
                    self._discord_bot, "is_in_call", False
                )
                if not in_call or self._turn_addressed_to_hermes:
                    logger.info("Despedida detectada: %r", self._user_turn_text.strip())
                    self._farewell_requested = True

        out_transcription = getattr(content, "output_transcription", None)
        out_text = (
            getattr(out_transcription, "text", None) if out_transcription else None
        )
        if out_text:
            self._assistant_turn_text += out_text

        model_turn = getattr(content, "model_turn", None)
        if model_turn is not None:
            in_call = self._discord_bot is not None and getattr(
                self._discord_bot, "is_in_call", False
            )
            is_creator_speaking = self._settings.discord_user_id is not None and (
                self._announced_speaker_id == self._settings.discord_user_id
                or self._call_speaker_id == self._settings.discord_user_id
            )
            # Transcrições podem chegar depois do primeiro áudio do modelo.
            has_foreign_text = (
                in_call
                and bool(self._user_turn_text.strip())
                and not self._turn_addressed_to_hermes
                and not is_creator_speaking
            )
            for part in model_turn.parts or []:
                inline = getattr(part, "inline_data", None)
                if inline is not None and inline.data:
                    self._turn_had_audio = True
                    if has_foreign_text:
                        self._pending_turn_audio.append(inline.data)
                    else:
                        if self._pending_turn_audio:
                            while self._pending_turn_audio:
                                p_chunk = self._pending_turn_audio.popleft()
                                self._audio_queue.put_nowait(p_chunk)
                                if in_call and self._discord_bot is not None:
                                    self._discord_bot.feed_audio(p_chunk)
                        self._audio_queue.put_nowait(inline.data)
                        if in_call and self._discord_bot is not None:
                            self._discord_bot.feed_audio(inline.data)

        if getattr(content, "turn_complete", False):
            in_call = self._discord_bot is not None and getattr(
                self._discord_bot, "is_in_call", False
            )
            is_creator_speaking = self._settings.discord_user_id is not None and (
                self._announced_speaker_id == self._settings.discord_user_id
                or self._call_speaker_id == self._settings.discord_user_id
            )
            has_foreign_text = (
                in_call
                and bool(self._user_turn_text.strip())
                and not self._turn_addressed_to_hermes
                and not is_creator_speaking
            )
            if self._pending_turn_audio:
                if not has_foreign_text:
                    while self._pending_turn_audio:
                        p_chunk = self._pending_turn_audio.popleft()
                        self._audio_queue.put_nowait(p_chunk)
                        if in_call and self._discord_bot is not None:
                            self._discord_bot.feed_audio(p_chunk)
                else:
                    logger.info(
                        "Resposta suprimida na call: %d chunks, transcrição=%r",
                        len(self._pending_turn_audio),
                        self._user_turn_text.strip(),
                    )
                    self._pending_turn_audio.clear()

            if in_call and self._discord_bot is not None:
                self._discord_bot.end_audio_turn()
                if self._turn_had_audio:
                    self._call_completed_reply = True

            u_clean = " ".join(self._user_turn_text.split())
            a_clean = " ".join(self._assistant_turn_text.split())

            if has_foreign_text:
                logger.debug(
                    "Conversa paralela na call ignorada (sem chamado 'Hermes'): %r",
                    u_clean,
                )
            else:
                if u_clean:
                    label = self._turn_speaker_name if in_call and self._turn_speaker_name else "Você"
                    print(f"[{label}] {u_clean}", flush=True)
                if a_clean:
                    print(f"[Ren] {a_clean}", flush=True)
                if u_clean or a_clean:
                    self._knowledge.record_turn(
                        u_clean, a_clean,
                        speaker_id=self._call_speaker_id if self._call_only else None,
                        speaker_name=self._turn_speaker_name if self._call_only else "",
                    )
                    self._maybe_schedule_compaction()
                if in_call and self._discord_bot is not None and (
                    self._chat_requested or self._turn_sources
                ):
                    message_parts = []
                    if a_clean:
                        message_parts.append(a_clean)
                    if self._turn_sources:
                        sources = "\n".join(
                            f"{title}: {uri}" for title, uri in self._turn_sources[:5]
                        )
                        message_parts.append(f"Fontes:\n{sources}")
                    if message_parts:
                        self._discord_bot.publish_text("\n\n".join(message_parts))
                if in_call and self._turn_had_audio and self._turn_addressed_to_hermes:
                    self._call_followup_until = time.monotonic() + _CALL_FOLLOWUP_SEC
                    self._call_idle_deadline = self._call_followup_until

            self._user_turn_text = ""
            self._assistant_turn_text = ""
            self._turn_had_audio = False
            self._turn_speaker_name = ""
            self._announced_speaker_id = None
            self._announced_speaker_name = ""
            self._turn_addressed_to_hermes = False
            self._pending_turn_audio.clear()
            self._screen_sent_in_turn = False
            self._logged_queries.clear()
            self._turn_sources.clear()
            self._chat_requested = False
            self._edge_router.reset_turn()
            if self._farewell_requested:
                # Encerra só depois de tocar a despedida do Hermes.
                self._audio_queue.put_nowait(_END_OF_TURN_SENTINEL)
            else:
                self._gateway.emit_state_changed(GatewayStatus.LISTENING)

    def _drain_audio_queue(self) -> None:
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def _play(self, out_stream: Any, executor: ThreadPoolExecutor) -> None:
        while not self._stop.is_set():
            try:
                data = await asyncio.wait_for(self._audio_queue.get(), _POLL_SEC)
            except asyncio.TimeoutError:
                continue
            if data is _END_OF_TURN_SENTINEL:
                if self._call_only and self._discord_bot is not None:
                    deadline = time.monotonic() + 10.0
                    while self._discord_bot.is_speaking and time.monotonic() < deadline:
                        await asyncio.sleep(0.05)
                reason = (
                    SessionEndReason.SHUTDOWN
                    if self._shutdown_requested
                    else SessionEndReason.LEFT_CALL
                    if self._leave_call_requested
                    else SessionEndReason.FAREWELL
                )
                self._finish(reason)
                return
            self._last_playback = time.monotonic()
            self._monitor.mark_activity()
            if self._gateway.current_status is not GatewayStatus.SPEAKING:
                self._gateway.emit_state_changed(GatewayStatus.SPEAKING)
            self._gateway.emit_audio_chunk_telemetry(data, sample_rate=OUTPUT_RATE)
            in_call = self._discord_bot is not None and getattr(
                self._discord_bot, "is_in_call", False
            )
            if in_call:
                # O Discord já reproduz o áudio em sua própria thread. Não atrase
                # a recepção dos próximos chunks simulando playback local.
                pass
            else:
                self._aec.feed_reference(data)
                await write_chunk(out_stream, data, executor)
            self._last_playback = time.monotonic()
            self._monitor.mark_activity()

    async def _announce_pending_reminders(self, session: Any) -> None:
        """Injeta aviso na sessão Live para o Hermes falar em voz alta qualquer lembrete disparado."""
        from google.genai import types

        while self._pending_reminders and not self._stop.is_set():
            label = self._pending_reminders.popleft()
            self._turn_addressed_to_hermes = True
            self._monitor.mark_activity()
            self._preroll_buffer.clear()
            prompt = (
                f"[SISTEMA - LEMBRETE PROGRAMADO DISPARADO AGORA]: O timer que o Vinícius "
                f"pediu acabou de zerar! Fale imediatamente em voz alta avisando o Vinícius "
                f"de forma direta e natural que está na hora de: '{label}'."
            )
            send_client = getattr(session, "send_client_content", None)
            if send_client is not None:
                try:
                    await send_client(
                        turns=types.Content(
                            role="user",
                            parts=[types.Part.from_text(text=prompt)],
                        ),
                        turn_complete=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Falha ao anunciar lembrete na sessão: %s", exc)

    async def _vad_monitor(self, session: Optional[Any] = None) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(_POLL_SEC)
            if self._call_only and (
                self._discord_bot is None or not self._discord_bot.is_in_call
            ):
                if self._leave_call_requested:
                    self._finish(SessionEndReason.LEFT_CALL)
                else:
                    logger.warning("Call desconectada durante a sessão; reconectando.")
                    self._finish(SessionEndReason.SERVER_CLOSED)
                return
            if (
                session is not None
                and self._pending_reminders
                and not self.assistant_is_speaking()
            ):
                await self._announce_pending_reminders(session)
            if self.assistant_is_speaking():
                self._monitor.mark_activity()
            else:
                in_call = self._discord_bot is not None and getattr(
                    self._discord_bot, "is_in_call", False
                )
                if in_call:
                    if self._call_only and self._call_completed_reply and self._call_handoff_checker is not None and self._call_handoff_checker():
                        self._finish(SessionEndReason.SILENCE)
                        return
                    if self._call_only and time.monotonic() >= self._call_idle_deadline:
                        self._finish(SessionEndReason.SILENCE)
                        return
                    self._monitor.mark_activity()
                    continue
                if self._monitor.is_timed_out(self._settings.vad_silence_timeout_sec):
                    logger.info(
                        "Silêncio por %.0fs: encerrando sessão.",
                        self._settings.vad_silence_timeout_sec,
                    )
                    self._finish(SessionEndReason.SILENCE)
                    return

    # ------------------------------------------------------------ supervisão
    async def _supervise(self, tasks: List["asyncio.Task[None]"]) -> None:
        stop_waiter = asyncio.create_task(self._stop.wait(), name="t_stop")
        try:
            await asyncio.wait(
                [*tasks, stop_waiter], return_when=asyncio.FIRST_COMPLETED
            )
            # Qualquer task que termine (ou falhe) encerra a sessão inteira.
            if not self._stop.is_set():
                self._finish(SessionEndReason.ERROR)
            # Dá tempo às tasks de I/O de saírem pelo próprio loop, sem
            # cancelar leituras/escritas do PyAudio no meio.
            pending = [t for t in tasks if not t.done()]
            if pending:
                await asyncio.wait(pending, timeout=_SHUTDOWN_GRACE_SEC)
        finally:
            for t in [*tasks, stop_waiter]:
                if not t.done():
                    t.cancel()
            results = await asyncio.gather(
                *tasks, stop_waiter, return_exceptions=True
            )
        for task, result in zip(tasks, results):
            if isinstance(result, Exception):
                if self._end_reason is not SessionEndReason.ERROR:
                    # Falha durante o desligamento de uma sessão já encerrada.
                    logger.debug("Task %s falhou ao encerrar: %r", task.get_name(), result)
                    continue
                logger.error("Task %s falhou: %r", task.get_name(), result)
                raise result

    async def run(self) -> SessionEndReason:
        """Executa uma sessão completa. Nunca propaga erros de rede/API."""
        self._reset_state()
        executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="hermes-audio")
        in_stream: Optional[Any] = None
        out_stream: Optional[Any] = None
        preroll_stop = asyncio.Event()
        preroll_task: Optional["asyncio.Task[None]"] = None
        try:
            if not self._call_only:
                in_stream = self._streams.open_input()
                out_stream = self._streams.open_output()
            # Começa a gravar o microfone IMEDIATAMENTE enquanto o WebSocket conecta.
            preroll_task = asyncio.create_task(
                self._capture_preroll(in_stream, executor, preroll_stop),
                name="t_preroll",
            )
            reconnect_attempts = 0
            first_connect = True
            while True:
                self._reconnect_requested = False
                if not first_connect:
                    self._stop.clear()
                    self._drain_audio_queue()
                try:
                    async with self._client.aio.live.connect(
                        model=self._settings.gemini_model, config=self.build_config()
                    ) as session:
                        if first_connect:
                            preroll_stop.set()
                            await preroll_task
                            logger.info(
                                "Conectado à Gemini Live API (%s). Chunks de pré-conexão: %d.",
                                self._settings.gemini_model,
                                len(self._preroll_buffer),
                            )
                            print("[Ren] Ouvindo...", flush=True)
                            self._gateway.emit_state_changed(GatewayStatus.LISTENING)
                            self._monitor.reset()
                            if self._pending_reminders:
                                await self._announce_pending_reminders(session)
                            first_connect = False
                        else:
                            logger.info(
                                "Sessão Gemini reconectada transparentemente (tentativa %d/%d).",
                                reconnect_attempts,
                                _MAX_RECONNECT_ATTEMPTS,
                            )
                            self._monitor.reset()
                        self._active_live_session = session
                        tasks = [
                            asyncio.create_task(
                                self._record_and_send(session, in_stream, executor),
                                name="t_rec",
                            ),
                            asyncio.create_task(self._receive(session), name="t_recv"),
                            asyncio.create_task(
                                self._play(out_stream, executor), name="t_play"
                            ),
                            asyncio.create_task(
                                self._vad_monitor(session), name="t_vad"
                            ),
                        ]
                        try:
                            await self._supervise(tasks)
                        finally:
                            self._active_live_session = None
                except asyncio.CancelledError:
                    if _current_task_is_cancelling():
                        raise  # Ctrl+C / cancelamento externo: deixa propagar.
                    logger.warning("Sessão cancelada internamente.")
                    self._end_reason = SessionEndReason.ERROR
                    break
                except Exception as exc:  # noqa: BLE001 - resiliência / reconexão
                    if (
                        _is_connection_closed_error(exc)
                        and self.resumption_handle
                        and reconnect_attempts < _MAX_RECONNECT_ATTEMPTS
                    ):
                        reconnect_attempts += 1
                        logger.warning(
                            "Conexão encerrada (%s); reconectando (%d/%d) com resumption_handle...",
                            exc,
                            reconnect_attempts,
                            _MAX_RECONNECT_ATTEMPTS,
                        )
                        await asyncio.sleep(0.2)
                        continue
                    raise

                if (
                    self._reconnect_requested
                    and self.resumption_handle
                    and reconnect_attempts < _MAX_RECONNECT_ATTEMPTS
                ):
                    reconnect_attempts += 1
                    logger.info(
                        "Reconexão transparente solicitada (%d/%d)...",
                        reconnect_attempts,
                        _MAX_RECONNECT_ATTEMPTS,
                    )
                    await asyncio.sleep(0.05)
                    continue
                break
        except asyncio.CancelledError:
            if _current_task_is_cancelling():
                raise  # Ctrl+C / cancelamento externo: deixa propagar.
            logger.warning("Sessão cancelada internamente.")
            self._end_reason = SessionEndReason.ERROR
        except Exception as exc:  # noqa: BLE001 - resiliência: volta ao standby
            logger.error(
                "Sessão Gemini encerrada por erro: %s",
                exc,
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
            self._end_reason = SessionEndReason.ERROR
        finally:
            preroll_stop.set()
            self._stop.set()
            self._gateway.emit_state_changed(GatewayStatus.STANDBY)
            if preroll_task is not None and not preroll_task.done():
                await asyncio.gather(preroll_task, return_exceptions=True)
            # Espera leituras/escritas em andamento antes de fechar os streams.
            executor.shutdown(wait=True)
            close_stream(in_stream)
            close_stream(out_stream)
            # Pipeline de consolidação pós-sessão (SQLite) em thread daemon
            threading.Thread(
                target=self._knowledge.consolidate_post_session,
                args=(self._client,),
                name="hermes-memory-consolidate",
                daemon=True,
            ).start()
        return self._end_reason
