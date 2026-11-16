"""Módulo de áudio do Discord: conversão bidirecional de áudio, AudioSource e AudioSink com suporte a DAVE E2EE."""

from __future__ import annotations

import logging
import platform
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import discord
import numpy as np

logger = logging.getLogger(__name__)

# Discord Voice: 48 kHz, 16-bit signed PCM, Stereo (2 canais), frames de 20 ms (960 amostras por canal)
DISCORD_SAMPLE_RATE: int = 48_000
DISCORD_CHANNELS: int = 2
DISCORD_FRAME_SAMPLES: int = 960  # 48000 * 0.02s
DISCORD_FRAME_BYTES: int = DISCORD_FRAME_SAMPLES * DISCORD_CHANNELS * 2  # 3840 bytes

# Hermes / Gemini Voice: 16 kHz, 16-bit signed PCM, Mono
HERMES_SAMPLE_RATE: int = 16_000

# Pacote oficial de silêncio Opus do Discord (3 bytes não criptografados pelo DAVE)
OPUS_SILENCE_PACKET: bytes = b"\xf8\xff\xfe"

try:
    import davey  # type: ignore[import-not-found]

    _HAS_DAVEY = True
except ImportError:
    davey = None  # type: ignore[assignment]
    _HAS_DAVEY = False


def ensure_opus_loaded() -> bool:
    """Carrega dinamicamente a DLL libopus oficial fornecida pelo discord.py no Windows."""
    try:
        from discord import opus

        if opus.is_loaded():
            return True

        if sys.platform == "win32":
            # Procura a DLL incluída no pacote discord.py
            is_64bit = "64" in platform.architecture()[0]
            dll_name = "libopus-0.x64.dll" if is_64bit else "libopus-0.x86.dll"
            pkg_bin = Path(discord.__file__).parent / "bin" / dll_name
            if pkg_bin.is_file():
                opus.load_opus(str(pkg_bin))
                return bool(opus.is_loaded())

            # Fallbacks no PATH
            for name in ("libopus-0.x64.dll", "libopus-0.dll", "opus.dll"):
                try:
                    opus.load_opus(name)
                    if opus.is_loaded():
                        return True
                except Exception:  # noqa: BLE001
                    pass
        return bool(opus.is_loaded())
    except Exception as exc:  # noqa: BLE001
        logger.warning("Falha ao inicializar biblioteca Opus do Discord: %s", exc)
        return False


def convert_24k_mono_to_48k_stereo(pcm_24k_mono: bytes) -> bytes:
    """Converte PCM 16-bit mono 24 kHz da Gemini Live em PCM 16-bit stereo 48 kHz do Discord.

    Cada amostra de 24 kHz é duplicada no tempo (2x upsampling para 48 kHz) e
    duplicada nos canais (Left e Right), resultando em 4 amostras int16 por amostra original.
    """
    usable = len(pcm_24k_mono) - (len(pcm_24k_mono) % 2)
    if usable < 2:
        return b""
    samples_24k = np.frombuffer(pcm_24k_mono[:usable], dtype=np.int16)
    # np.repeat 4x: [s0_L, s0_R, s0_L, s0_R, s1_L, s1_R, s1_L, s1_R...]
    samples_48k_stereo = np.repeat(samples_24k, 4)
    return samples_48k_stereo.tobytes()


def convert_48k_stereo_to_16k_mono(pcm_48k_stereo: bytes) -> bytes:
    """Converte PCM 16-bit stereo 48 kHz do Discord em PCM 16-bit mono 16 kHz para o Hermes/Gemini."""
    usable = len(pcm_48k_stereo) - (len(pcm_48k_stereo) % 4)
    if usable < 4:
        return b""
    samples = np.frombuffer(pcm_48k_stereo[:usable], dtype=np.int16)
    left = samples[0::2].astype(np.int32)
    right = samples[1::2].astype(np.int32)
    mono = (left + right) // 2
    rem = len(mono) % 3
    if rem > 0:
        mono = mono[:-rem]
    if len(mono) < 3:
        return b""
    # Downsample de 48 kHz para 16 kHz (média de cada trio de amostras)
    downsampled = (mono[0::3] + mono[1::3] + mono[2::3]) // 3
    return downsampled.astype(np.int16).tobytes()


def mix_pcm_16k(pcm_a: bytes, pcm_b: bytes) -> bytes:
    """Mistura dois fluxos PCM 16 kHz mono com saturação (clipping) suave."""
    if not pcm_a:
        return pcm_b
    if not pcm_b:
        return pcm_a
    min_len = min(len(pcm_a), len(pcm_b))
    min_len -= min_len % 2
    s_a = np.frombuffer(pcm_a[:min_len], dtype=np.int16).astype(np.int32)
    s_b = np.frombuffer(pcm_b[:min_len], dtype=np.int16).astype(np.int32)
    mixed = np.clip(s_a + s_b, -32768, 32767).astype(np.int16).tobytes()
    if len(pcm_a) > min_len:
        mixed += pcm_a[min_len:]
    elif len(pcm_b) > min_len:
        mixed += pcm_b[min_len:]
    return mixed


class HermesDiscordAudioSource(discord.AudioSource):
    """AudioSource thread-safe do Discord que consome buffers de fala do Hermes em tempo real.

    Emite 3840 bytes (20 ms a 48 kHz stereo) continuamente sem engasgos e expõe
    `is_speaking_active` para gerenciamento automático de SpeakingState (pause/resume) no Discord.
    """

    def __init__(self, max_buffer_bytes: int = 48_000 * 4 * 120, startup_frames: int = 0) -> None:  # até 120s de fala
        super().__init__()
        self._lock = threading.Lock()
        self._buffer = bytearray()
        self._max_buffer = max_buffer_bytes
        self._silence_frame = b"\x00" * DISCORD_FRAME_BYTES
        self._has_spoken = False
        self._tail_frames: int = 0
        self._startup_frames = max(0, startup_frames)
        self._startup_remaining = 0
        self._played_audio = False
        self._underrun_reported = False
        self._turn_open = False

    def is_opus(self) -> bool:
        return False

    @property
    def is_speaking_active(self) -> bool:
        """True enquanto há áudio no buffer ou dentro da janela de cauda (400 ms) entre chunks."""
        with self._lock:
            return self._turn_open or len(self._buffer) > 0 or self._tail_frames > 0

    def feed(self, pcm_24k_mono: bytes) -> None:
        """Recebe chunks de 24 kHz mono do Gemini Live e enfileira como 48 kHz stereo."""
        pcm_48k = convert_24k_mono_to_48k_stereo(pcm_24k_mono)
        if not pcm_48k:
            return
        with self._lock:
            self._turn_open = True
            if self._underrun_reported:
                logger.warning("Discord: áudio do Gemini retomou após esvaziar o buffer de saída")
                self._underrun_reported = False
            if not self._buffer and self._tail_frames == 0:
                self._startup_remaining = self._startup_frames
            capacity = max(DISCORD_FRAME_BYTES, self._max_buffer)
            capacity -= capacity % DISCORD_FRAME_BYTES
            self._buffer.extend(pcm_48k)
            if len(self._buffer) > capacity:
                excess = len(self._buffer) - capacity
                drop = ((excess + DISCORD_FRAME_BYTES - 1) // DISCORD_FRAME_BYTES) * DISCORD_FRAME_BYTES
                del self._buffer[:drop]
                logger.warning("Discord: buffer de saída cheio; %.0f ms de áudio descartados", drop / DISCORD_FRAME_BYTES * 20)
            self._has_spoken = True
            self._tail_frames = 20  # 20 frames * 20 ms = 400 ms de tolerância entre chunks

    def end_turn(self) -> None:
        """Permite pausar o player somente após o último chunk da resposta."""
        with self._lock:
            self._turn_open = False

    def read(self) -> bytes:
        """Chamado pelo encoder do Discord a cada 20 ms (50 vezes por segundo)."""
        with self._lock:
            if self._startup_remaining > 0:
                self._startup_remaining -= 1
                return self._silence_frame
            if len(self._buffer) >= DISCORD_FRAME_BYTES:
                chunk = bytes(self._buffer[:DISCORD_FRAME_BYTES])
                del self._buffer[:DISCORD_FRAME_BYTES]
                self._tail_frames = 20
                self._played_audio = True
                self._underrun_reported = False
                return chunk
            if len(self._buffer) > 0:
                rem = bytes(self._buffer)
                self._buffer.clear()
                self._played_audio = True
                self._underrun_reported = False
                return rem + b"\x00" * (DISCORD_FRAME_BYTES - len(rem))
            if self._tail_frames > 0:
                if self._turn_open and self._played_audio and not self._underrun_reported:
                    self._underrun_reported = True
                self._tail_frames -= 1
            return self._silence_frame

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()
            self._tail_frames = 0
            self._startup_remaining = 0
            self._played_audio = False
            self._underrun_reported = False
            self._turn_open = False

    def cleanup(self) -> None:
        self.clear()


# Importação segura de voice_recv.AudioSink se disponível
try:
    import discord.ext.voice_recv as voice_recv

    _BaseAudioSink = voice_recv.AudioSink
except Exception:  # noqa: BLE001
    _BaseAudioSink = object  # type: ignore[assignment,misc]


class HermesDiscordVoiceReceiver(_BaseAudioSink):
    """AudioSink do discord-ext-voice-recv com descriptografia DAVE E2EE e mixagem multi-usuário por SSRC.

    1. Opera com wants_opus() = True para interceptar o payload antes da decodificação Opus.
    2. Descriptografa a camada E2EE DAVE (`dave_session.decrypt`) introduzida pelo Discord/discord.py 2.7+.
    3. Mantém um decodificador Opus e um Jitter Buffer separado por SSRC (orador), mixando
       múltiplas pessoas falando simultaneamente sem desacelerar ou picotar o áudio.
    """

    def __init__(self, max_buffer_bytes: int = 16_000 * 2 * 2) -> None:  # 2 segundos máx por SSRC
        if _BaseAudioSink is not object:
            super().__init__()
        self._lock = threading.Lock()
        self._ssrc_buffers: Dict[int, bytearray] = {}
        self._speaker_buffers: Dict[int, bytearray] = {}
        self._wake_buffers: Dict[int, bytearray] = {}
        self._speaker_names: Dict[int, str] = {}
        self._ssrc_last_time: Dict[int, float] = {}
        self._ssrc_streaming: set[int] = set()
        self._ssrc_to_uid: Dict[int, int] = {}
        self._decoders: Dict[int, Any] = {}
        self._max_buffer = max_buffer_bytes
        self._speaker_max_buffer = max(max_buffer_bytes, 16_000 * 2 * 10)
        self._last_speech_time: float = 0.0
        self._last_speaker_name: str = ""
        self._last_speaker_username: str = ""
        self._last_speaker_id: Optional[int] = None
        self._vc_ref: Optional[Any] = None

    def set_voice_client(self, vc: Optional[Any]) -> None:
        """Associa explicitamente o VoiceRecvClient ativo para acesso à sessão DAVE e mapa SSRC."""
        self._vc_ref = vc

    def wants_opus(self) -> bool:
        return True

    def _resolve_voice_client(self) -> Optional[Any]:
        if self._vc_ref is not None:
            return self._vc_ref
        return getattr(self, "voice_client", None)

    def _decrypt_dave_if_needed(
        self, vc: Optional[Any], ssrc: int, user: Optional[Any], opus_bytes: bytes
    ) -> Optional[bytes]:
        """Remove a camada de criptografia ponta-a-ponta DAVE (MLS) se ativa na chamada."""
        if not _HAS_DAVEY or davey is None or vc is None:
            return opus_bytes

        conn = getattr(vc, "_connection", None)
        if conn is None:
            return opus_bytes

        dave_session = getattr(conn, "dave_session", None)
        dave_ver = getattr(conn, "dave_protocol_version", 0)
        if dave_session is None or dave_ver <= 0:
            return opus_bytes

        # 1. Tenta obter o user_id diretamente do objeto user, do VoiceRecvClient ou do cache local
        uid: Optional[int] = getattr(user, "id", None)
        if uid is None and hasattr(vc, "_get_id_from_ssrc"):
            uid = vc._get_id_from_ssrc(ssrc)
        if uid is None:
            with self._lock:
                uid = self._ssrc_to_uid.get(ssrc)

        if uid is not None:
            try:
                decrypted = dave_session.decrypt(int(uid), davey.MediaType.audio, opus_bytes)
                with self._lock:
                    self._ssrc_to_uid[ssrc] = int(uid)
                return decrypted
            except Exception:  # noqa: BLE001
                try:
                    if dave_session.can_passthrough(int(uid)):
                        return opus_bytes
                except Exception:  # noqa: BLE001
                    pass
                return None

        # 2. Caso o bot tenha entrado com alguém já falando (sem evento op=5 SPEAKING prévio),
        # testa os user_ids presentes na sessão DAVE / estado de voz do servidor e faz cache do SSRC
        candidate_uids: List[int] = []
        try:
            candidate_uids.extend(int(u) for u in dave_session.get_user_ids())
        except Exception:  # noqa: BLE001
            pass
        guild = getattr(vc, "guild", None)
        if guild is not None:
            for u_id in getattr(guild, "_voice_states", {}).keys():
                if int(u_id) not in candidate_uids:
                    candidate_uids.append(int(u_id))

        bot_uid = getattr(getattr(vc, "user", None), "id", None)
        for cand_uid in candidate_uids:
            if bot_uid is not None and cand_uid == int(bot_uid):
                continue
            try:
                decrypted = dave_session.decrypt(cand_uid, davey.MediaType.audio, opus_bytes)
                with self._lock:
                    self._ssrc_to_uid[ssrc] = cand_uid
                if hasattr(vc, "_add_ssrc"):
                    try:
                        vc._add_ssrc(cand_uid, ssrc)
                    except Exception:  # noqa: BLE001
                        pass
                return decrypted
            except Exception:  # noqa: BLE001
                continue

        return None

    def write(self, user: Optional[Any], data: Any) -> None:
        """Callback executado pela thread de recepção para cada pacote de áudio recebido."""
        if user is not None and getattr(user, "bot", False):
            return

        ssrc: int = int(getattr(getattr(data, "packet", None), "ssrc", 0) or 0)
        opus_bytes: Optional[bytes] = getattr(data, "opus", None)

        if not opus_bytes:
            # Fallback caso pcm já venha decodificado (ex.: testes unitários com mock)
            pcm_48k = getattr(data, "pcm", None)
        else:
            if opus_bytes == OPUS_SILENCE_PACKET:
                # Pacote de silêncio puro do Discord: não precisa poluir os buffers de voz
                return

            vc = self._resolve_voice_client()
            decrypted_opus = self._decrypt_dave_if_needed(vc, ssrc, user, opus_bytes)
            if not decrypted_opus:
                return

            with self._lock:
                if ssrc not in self._decoders:
                    try:
                        self._decoders[ssrc] = discord.opus.Decoder()
                    except Exception:  # noqa: BLE001
                        return
                decoder = self._decoders[ssrc]

            try:
                pcm_48k = decoder.decode(decrypted_opus)
            except Exception:  # noqa: BLE001
                return

        if not pcm_48k:
            return

        pcm_16k = convert_48k_stereo_to_16k_mono(pcm_48k)
        if not pcm_16k:
            return

        now = time.monotonic()
        name = (
            getattr(user, "display_name", None)
            or getattr(user, "name", None)
            or f"Membro-{ssrc}"
        )

        with self._lock:
            self._last_speech_time = now
            self._last_speaker_name = str(name)
            self._last_speaker_username = str(getattr(user, "name", "") or "")
            self._last_speaker_id = getattr(user, "id", None)
            speaker_id = getattr(user, "id", None) or self._ssrc_to_uid.get(ssrc)
            if speaker_id is None:
                vc = self._resolve_voice_client()
                if vc is not None and hasattr(vc, "_get_id_from_ssrc"):
                    try:
                        speaker_id = vc._get_id_from_ssrc(ssrc)
                    except Exception:  # noqa: BLE001
                        speaker_id = None
            if isinstance(speaker_id, int) and not isinstance(speaker_id, bool):
                self._speaker_names[speaker_id] = str(name)
                for buffers in (self._speaker_buffers, self._wake_buffers):
                    speaker_buf = buffers.setdefault(speaker_id, bytearray())
                    speaker_buf.extend(pcm_16k)
                    limit = self._speaker_max_buffer if buffers is self._speaker_buffers else self._max_buffer
                    if len(speaker_buf) > limit:
                        del speaker_buf[:len(speaker_buf) - limit]
            buf = self._ssrc_buffers.get(ssrc)
            if buf is None:
                buf = bytearray()
                self._ssrc_buffers[ssrc] = buf
            self._ssrc_last_time[ssrc] = now

            if len(buf) + len(pcm_16k) <= self._max_buffer:
                buf.extend(pcm_16k)
            else:
                excess = (len(buf) + len(pcm_16k)) - self._max_buffer
                del buf[:excess]
                buf.extend(pcm_16k)

    def read_chunk(self, max_bytes: int) -> bytes:
        """Extrai até max_bytes mixando todos os oradores ativos na call sem buracos de 4 ms ou 64 ms."""
        now = time.monotonic()
        mixed_chunk = b""

        with self._lock:
            empty_ssrcs: List[int] = []
            for ssrc, buf in self._ssrc_buffers.items():
                if not buf:
                    empty_ssrcs.append(ssrc)
                    self._ssrc_streaming.discard(ssrc)
                    continue

                last_t = self._ssrc_last_time.get(ssrc, 0.0)
                if (now - last_t) > 1.8:
                    # Descarta sobras antigas de falas já encerradas há mais de 1.8s
                    buf.clear()
                    empty_ssrcs.append(ssrc)
                    self._ssrc_streaming.discard(ssrc)
                    continue

                is_actively_streaming = (now - last_t) < 0.15

                if len(buf) >= max_bytes:
                    self._ssrc_streaming.add(ssrc)
                    part = bytes(buf[:max_bytes])
                    del buf[:max_bytes]
                    if not buf:
                        empty_ssrcs.append(ssrc)
                elif is_actively_streaming:
                    if ssrc in self._ssrc_streaming and len(buf) >= (max_bytes * 3) // 4:
                        # Temos ex.: 1920 bytes (3 pacotes de 20 ms = 60 ms) quando max_bytes é 2048 (64 ms).
                        # Interpola suavemente para max_bytes para não pular o ciclo nem criar buraco de 4 ms.
                        raw_part = bytes(buf)
                        buf.clear()
                        empty_ssrcs.append(ssrc)
                        n_target = max_bytes // 2
                        src_samples = np.frombuffer(
                            raw_part[: len(raw_part) - (len(raw_part) % 2)], dtype=np.int16
                        ).astype(np.float32)
                        if len(src_samples) >= 2 and n_target > 0:
                            x_old = np.linspace(0.0, 1.0, num=len(src_samples), endpoint=False)
                            x_new = np.linspace(0.0, 1.0, num=n_target, endpoint=False)
                            part = np.interp(x_new, x_old, src_samples).astype(np.int16).tobytes()
                        else:
                            part = raw_part
                    else:
                        # Início da frase: aguarda acumular ao menos 1 chunk completo (~64 ms)
                        self._ssrc_streaming.discard(ssrc)
                        continue
                else:
                    # Fim da frase: drena o restante imediatamente
                    self._ssrc_streaming.discard(ssrc)
                    count = min(len(buf), max_bytes)
                    part = bytes(buf[:count])
                    del buf[:count]
                    if not buf:
                        empty_ssrcs.append(ssrc)

                mixed_chunk = mix_pcm_16k(mixed_chunk, part) if mixed_chunk else part

            for ssrc in empty_ssrcs:
                self._ssrc_buffers.pop(ssrc, None)

        return mixed_chunk

    def read_speaker_chunk(self, speaker_id: int, max_bytes: int) -> bytes:
        """Lê apenas a voz de um membro, sem consumir o fluxo do detector."""
        with self._lock:
            buf = self._speaker_buffers.get(speaker_id)
            if not buf:
                return b""
            count = min(len(buf), max_bytes)
            chunk = bytes(buf[:count])
            del buf[:count]
            return chunk

    def read_wake_chunks(self, max_bytes: int) -> Dict[int, bytes]:
        """Lê separadamente o áudio disponível de cada membro para a wake word."""
        with self._lock:
            chunks: Dict[int, bytes] = {}
            for speaker_id, buf in self._wake_buffers.items():
                if buf:
                    count = min(len(buf), max_bytes)
                    chunks[speaker_id] = bytes(buf[:count])
                    del buf[:count]
            return chunks

    def speaker_name(self, speaker_id: int) -> str:
        with self._lock:
            return self._speaker_names.get(speaker_id, "")

    def clear_speaker_audio(self, speaker_id: int) -> None:
        with self._lock:
            self._speaker_buffers.pop(speaker_id, None)

    def clear_wake_audio(self) -> None:
        with self._lock:
            self._wake_buffers.clear()

    def reset_call_input(self) -> None:
        """Remove áudio antigo antes de uma nova escuta da wake word."""
        with self._lock:
            self._speaker_buffers.clear()
            self._wake_buffers.clear()

    def has_audio(self) -> bool:
        with self._lock:
            return any(len(buf) > 0 for buf in self._ssrc_buffers.values())

    @property
    def last_speech_time(self) -> float:
        return self._last_speech_time

    @property
    def last_speaker_name(self) -> str:
        return self._last_speaker_name

    @property
    def last_speaker_username(self) -> str:
        return self._last_speaker_username

    @property
    def last_speaker_id(self) -> Optional[int]:
        return self._last_speaker_id

    def clear(self) -> None:
        with self._lock:
            self._ssrc_buffers.clear()
            self._speaker_buffers.clear()
            self._wake_buffers.clear()
            self._speaker_names.clear()
            self._ssrc_last_time.clear()
            self._ssrc_streaming.clear()
            self._ssrc_to_uid.clear()
            self._decoders.clear()

    def cleanup(self) -> None:
        self.clear()
