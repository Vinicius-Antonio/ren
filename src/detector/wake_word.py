"""Wrapper de escuta da wake word (Porcupine ou Vosk PT-BR offline sem chave, com fallbacks).

Ordem de preferência:
1. Porcupine + modelo customizado `models/hermes.ppn` (se `PICOVOICE_ACCESS_KEY` definida).
2. Porcupine + keyword built-in (`WAKE_WORD_BUILTIN`, se `PICOVOICE_ACCESS_KEY` definida).
3. Vosk PT-BR offline (`models/vosk-model-small-pt-0.3`) — 100% gratuito, sem conta/chave,
   reconhecendo a palavra "Hermes" em português diretamente na CPU.
4. Teclado (Enter) — fallback final quando não há chave Picovoice nem modelo Vosk local.

Tudo roda localmente: nenhuma requisição de rede é feita durante a escuta.
"""

from __future__ import annotations

import enum
import time
import json
import logging
import threading
from collections import deque
from typing import Any, Callable, Optional

import numpy as np

from src.audio.stream_manager import AudioStreamManager, close_stream
from src.config import CHUNK_SIZE, INPUT_RATE, Settings

logger = logging.getLogger(__name__)


class CallDisconnectedError(RuntimeError):
    """A chamada caiu enquanto a wake word era aguardada."""

_WAKE_ALIASES = frozenset({"ren", "rem", "rei", "hein", "hem", "rende", "renda", "hermes"})

_VOSK_GRAMMAR = json.dumps(
    [
        "rei",
        "hein",
        "hem",
        "ei",
        "rende",
        "renda",
        "hermes",
        "oi hermes",
        "ei hermes",
        "fala hermes",
        "bom dia hermes",
        "boa tarde hermes",
        "boa noite hermes",
        "desligar hermes",
        "encerrar sistema",
        "[unk]",
    ]
)


class WakeWordMode(enum.Enum):
    CUSTOM_MODEL = "custom_model"
    BUILTIN_KEYWORD = "builtin_keyword"
    VOSK_PT = "vosk_pt"
    KEYBOARD = "keyboard"


def _load_porcupine() -> Any:
    import pvporcupine

    return pvporcupine


def _load_vosk() -> Any:
    import vosk  # type: ignore[import-not-found]

    vosk.SetLogLevel(-1)
    return vosk


class WakeWordDetector:
    def __init__(
        self,
        settings: Settings,
        stream_manager: AudioStreamManager,
        porcupine_module: Optional[Any] = None,
        input_fn: Callable[[str], str] = input,
        vosk_module: Optional[Any] = None,
        discord_bot: Optional[Any] = None,
        call_only: bool = False,
    ) -> None:
        self._settings = settings
        self._streams = stream_manager
        self._input_fn = input_fn
        self._discord_bot = discord_bot
        self._call_only = call_only
        self._porcupine: Optional[Any] = None
        self._vosk_rec: Optional[Any] = None
        self._vosk_module: Optional[Any] = None
        self._vosk_model: Optional[Any] = None
        self._speaker_recognizers: dict[int, Any] = {}
        self.wake_speaker_id: Optional[int] = None
        self._pending_call_speakers: deque[tuple[int, float]] = deque()
        self._pending_lock = threading.Lock()
        self._call_monitor_stop = threading.Event()
        self._call_monitor_thread: Optional[threading.Thread] = None
        self._external_wakeup = threading.Event()
        self.mode: WakeWordMode
        self.label: str

        if call_only and not settings.vosk_model_path.is_dir():
            raise ValueError("Modo call com identificação por pessoa requer o modelo Vosk local.")

        if not settings.picovoice_access_key or (call_only and settings.vosk_model_path.is_dir()):
            if settings.vosk_model_path.is_dir():
                try:
                    vk = vosk_module or _load_vosk()
                    model = vk.Model(str(settings.vosk_model_path))
                    self._vosk_module = vk
                    self._vosk_model = model
                    self._vosk_rec = vk.KaldiRecognizer(
                        model, INPUT_RATE, _VOSK_GRAMMAR
                    )
                    self.mode = WakeWordMode.VOSK_PT
                    self.label = "Ren"
                    logger.info(
                        "Detector Vosk PT-BR ativo (%s): diga 'Ren'.",
                        settings.vosk_model_path.name,
                    )
                    return
                except Exception as exc:  # noqa: BLE001
                    if call_only:
                        raise RuntimeError("Falha ao iniciar Vosk por participante na call") from exc
                    logger.warning(
                        "Falha ao iniciar Vosk (%s); usando ativação por teclado.",
                        exc,
                    )

            self.mode = WakeWordMode.KEYBOARD
            self.label = "tecla Enter"
            logger.warning(
                "Sem PICOVOICE_ACCESS_KEY ou modelo Vosk: usando ativação por teclado (Enter)."
            )
            return

        pv = porcupine_module or _load_porcupine()
        sensitivity = settings.wake_word_sensitivity
        if settings.wake_word_model_path.is_file():
            create_kwargs: dict[str, Any] = {
                "access_key": settings.picovoice_access_key,
                "keyword_paths": [str(settings.wake_word_model_path)],
                "sensitivities": [sensitivity],
            }
            if settings.wake_word_language_model_path.is_file():
                create_kwargs["model_path"] = str(
                    settings.wake_word_language_model_path
                )
            try:
                self._porcupine = pv.create(**create_kwargs)
                self.mode = WakeWordMode.CUSTOM_MODEL
                self.label = settings.wake_word_model_path.stem.capitalize()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Falha ao carregar %s (%s). Se o modelo foi treinado em português, "
                    "coloque 'porcupine_params_pt.pv' em %s. Usando keyword built-in '%s'.",
                    settings.wake_word_model_path,
                    exc,
                    settings.wake_word_language_model_path,
                    settings.wake_word_builtin,
                )
                self._init_builtin(pv, settings, sensitivity)
        else:
            self._init_builtin(pv, settings, sensitivity)
            logger.warning(
                "Modelo %s não encontrado: usando keyword built-in '%s'.",
                settings.wake_word_model_path,
                settings.wake_word_builtin,
            )

        assert self._porcupine is not None
        if self._porcupine.sample_rate != INPUT_RATE:
            raise RuntimeError(
                f"Porcupine espera {self._porcupine.sample_rate} Hz; "
                f"o Hermes captura em {INPUT_RATE} Hz."
            )

    def _init_builtin(
        self, pv: Any, settings: Settings, sensitivity: float
    ) -> None:
        self._porcupine = pv.create(
            access_key=settings.picovoice_access_key,
            keywords=[settings.wake_word_builtin],
            sensitivities=[sensitivity],
        )
        self.mode = WakeWordMode.BUILTIN_KEYWORD
        self.label = settings.wake_word_builtin

    def trigger_wakeup(self) -> None:
        """Acorda o detector imediatamente a partir de outra thread (ex.: disparo de Timer/Lembrete)."""
        self._external_wakeup.set()

    def wait_for_wake_word(self) -> None:
        """Bloqueia até a wake word ser detectada (ou um Timer/Lembrete acordar o Hermes).

        O stream do microfone é SEMPRE fechado antes de retornar, liberando o
        dispositivo para a sessão do Gemini.
        """
        self.wake_speaker_id = None
        if self._call_only:
            with self._pending_lock:
                while self._pending_call_speakers:
                    speaker_id, detected_at = self._pending_call_speakers.popleft()
                    if time.monotonic() - detected_at < 10.0:
                        self.wake_speaker_id = speaker_id
                        return
        if self._external_wakeup.is_set():
            self._external_wakeup.clear()
            return

        if self.mode is WakeWordMode.KEYBOARD:
            self._input_fn("")
            return

        if self.mode is WakeWordMode.VOSK_PT:
            self._wait_vosk()
            return

        assert self._porcupine is not None
        frame_length: int = self._porcupine.frame_length
        stream = None
        try:
            if not self._call_only:
                stream = self._streams.open_input(
                    rate=INPUT_RATE, frames_per_buffer=frame_length
                )
            while True:
                if self._call_only and not self._discord_bot.is_in_call:
                    raise CallDisconnectedError("Bot desconectado da call")
                if self._external_wakeup.is_set():
                    self._external_wakeup.clear()
                    return
                if self._call_only:
                    time.sleep(frame_length / INPUT_RATE)
                    raw = self._discord_bot.get_incoming_chunk(frame_length * 2)
                    raw = raw.ljust(frame_length * 2, b"\x00")
                else:
                    raw = stream.read(frame_length, exception_on_overflow=False)
                pcm = np.frombuffer(raw, dtype=np.int16)
                if self._porcupine.process(pcm.tolist()) >= 0:
                    return
        finally:
            close_stream(stream)

    def _wait_vosk(self) -> None:
        assert self._vosk_rec is not None
        if self._call_only:
            self._wait_vosk_call()
            return
        self._vosk_rec.Reset()
        stream = None
        try:
            if not self._call_only:
                stream = self._streams.open_input(
                    rate=INPUT_RATE, frames_per_buffer=CHUNK_SIZE * 2
                )
            while True:
                if self._call_only and not self._discord_bot.is_in_call:
                    raise CallDisconnectedError("Bot desconectado da call")
                if self._external_wakeup.is_set():
                    self._external_wakeup.clear()
                    return
                if self._call_only:
                    time.sleep(CHUNK_SIZE * 2 / INPUT_RATE)
                    raw = self._discord_bot.get_incoming_chunk(CHUNK_SIZE * 4)
                    raw = raw.ljust(CHUNK_SIZE * 4, b"\x00")
                else:
                    raw = stream.read(CHUNK_SIZE * 2, exception_on_overflow=False)
                if self._vosk_rec.AcceptWaveform(raw):
                    res = json.loads(self._vosk_rec.Result())
                    text = res.get("text", "")
                else:
                    part = json.loads(self._vosk_rec.PartialResult())
                    text = part.get("partial", "")
                words = text.lower().split()
                if ("desligar" in words and any(w in words for w in _WAKE_ALIASES)) or (
                    "encerrar" in words and "sistema" in words
                ):
                    self._vosk_rec.Reset()
                    raise KeyboardInterrupt("Comando de voz de encerramento em standby")
                if any(w in words for w in _WAKE_ALIASES):
                    self._vosk_rec.Reset()
                    return
        finally:
            close_stream(stream)

    def _wait_vosk_call(self) -> None:
        """Detecta o chamado por ID, sem atribuir o áudio misturado ao último falante."""
        if not self._discord_bot.is_in_call:
            raise CallDisconnectedError("Bot desconectado da call")
        assert self._vosk_module is not None and self._vosk_model is not None
        self._speaker_recognizers.clear()
        self._discord_bot.reset_call_input()
        while True:
            if not self._discord_bot.is_in_call:
                raise CallDisconnectedError("Bot desconectado da call")
            if self._external_wakeup.is_set():
                self._external_wakeup.clear()
                return
            time.sleep(CHUNK_SIZE / INPUT_RATE)
            for speaker_id, raw in self._discord_bot.get_wake_chunks(CHUNK_SIZE * 4).items():
                rec = self._speaker_recognizers.get(speaker_id)
                if rec is None:
                    rec = self._vosk_module.KaldiRecognizer(
                        self._vosk_model, INPUT_RATE, _VOSK_GRAMMAR
                    )
                    self._speaker_recognizers[speaker_id] = rec
                complete = rec.AcceptWaveform(raw)
                result = json.loads(rec.Result() if complete else rec.PartialResult())
                words = str(result.get("text" if complete else "partial", "")).lower().split()
                if any(word in _WAKE_ALIASES for word in words):
                    self.wake_speaker_id = speaker_id
                    return

    def start_call_monitor(self) -> None:
        """Continua ouvindo chamados de outros membros durante a resposta ativa."""
        if not self._call_only or self._vosk_module is None or self._vosk_model is None:
            return
        self._call_monitor_stop.clear()
        self._discord_bot.clear_wake_audio()

        def monitor() -> None:
            recognizers: dict[int, Any] = {}
            while not self._call_monitor_stop.wait(CHUNK_SIZE / INPUT_RATE):
                if not self._discord_bot.is_in_call:
                    return
                for speaker_id, raw in self._discord_bot.get_wake_chunks(CHUNK_SIZE * 4).items():
                    if speaker_id == self.wake_speaker_id:
                        continue
                    rec = recognizers.get(speaker_id)
                    if rec is None:
                        rec = self._vosk_module.KaldiRecognizer(self._vosk_model, INPUT_RATE, _VOSK_GRAMMAR)
                        recognizers[speaker_id] = rec
                    complete = rec.AcceptWaveform(raw)
                    result = json.loads(rec.Result() if complete else rec.PartialResult())
                    words = str(result.get("text" if complete else "partial", "")).lower().split()
                    if any(word in _WAKE_ALIASES for word in words):
                        with self._pending_lock:
                            if not any(uid == speaker_id for uid, _ in self._pending_call_speakers):
                                self._pending_call_speakers.append((speaker_id, time.monotonic()))
                        rec.Reset()

        self._call_monitor_thread = threading.Thread(target=monitor, name="ren-call-wake", daemon=True)
        self._call_monitor_thread.start()

    def stop_call_monitor(self) -> None:
        self._call_monitor_stop.set()
        if self._call_monitor_thread is not None:
            self._call_monitor_thread.join(timeout=1.0)
            self._call_monitor_thread = None

    def has_pending_call_speaker(self) -> bool:
        with self._pending_lock:
            return any(time.monotonic() - at < 10.0 for _, at in self._pending_call_speakers)

    def close(self) -> None:
        self.stop_call_monitor()
        if self._porcupine is not None:
            try:
                self._porcupine.delete()
            finally:
                self._porcupine = None
        self._vosk_rec = None
