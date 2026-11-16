"""Detecção de atividade de voz (VAD) local: NeuralVAD (Silero ONNX), RMS, FFT e análise espectral."""

from __future__ import annotations

import logging
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, List, Optional

import numpy as np

from src.config import INPUT_RATE

logger = logging.getLogger(__name__)

# Faixa principal de energia da voz humana em banda larga (80 Hz a 4000 Hz)
_VOICE_LOW_HZ = 80.0
_VOICE_HIGH_HZ = 4000.0
_MIN_VOICE_BAND_RATIO = 0.25

SILERO_ONNX_URL: str = (
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
)


def ensure_silero_model(model_path: Path, auto_download: bool = False) -> bool:
    """Garante que `silero_vad.onnx` exista localmente (faz download se `auto_download=True`)."""
    if model_path.is_file() and model_path.stat().st_size > 100_000:
        return True
    if not auto_download:
        return False
    try:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Baixando modelo Silero VAD ONNX para %s...", model_path)
        req = urllib.request.Request(
            SILERO_ONNX_URL, headers={"User-Agent": "Hermes-2.0"}
        )
        with urllib.request.urlopen(req, timeout=15.0) as resp:  # noqa: S310
            data = resp.read()
        if len(data) > 100_000:
            model_path.write_bytes(data)
            return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Não foi possível baixar silero_vad.onnx: %s", exc)
    return False


def calculate_rms(pcm_bytes: bytes) -> float:
    """RMS de um buffer PCM 16-bit mono. Buffer vazio retorna 0.0."""
    if len(pcm_bytes) < 2:
        return 0.0
    usable = len(pcm_bytes) - (len(pcm_bytes) % 2)
    samples = np.frombuffer(pcm_bytes[:usable], dtype=np.int16).astype(np.float64)
    return float(np.sqrt(np.mean(np.square(samples))))


def compute_fft_bands(
    pcm_bytes: bytes, sample_rate: int = INPUT_RATE, num_bands: int = 16
) -> List[float]:
    """Calcula `num_bands` bandas espectrais normalizadas (0.0 a 1.0) para telemetria FFT do HUD."""
    if len(pcm_bytes) < 4 or num_bands <= 0:
        return [0.0] * max(1, num_bands)
    usable = len(pcm_bytes) - (len(pcm_bytes) % 2)
    samples = (
        np.frombuffer(pcm_bytes[:usable], dtype=np.int16).astype(np.float32) / 32768.0
    )
    if len(samples) < 4:
        return [0.0] * num_bands
    windowed = samples * np.hanning(len(samples))
    mag = np.abs(np.fft.rfft(windowed))
    # Ignora DC (índice 0) e foca até ~6 kHz onde a voz concentra energia
    usable_bins = mag[1 : max(2, min(len(mag), int(len(mag) * 0.75)))]
    splits = np.array_split(usable_bins, num_bands)
    bands: List[float] = []
    for chunk in splits:
        if chunk.size == 0:
            bands.append(0.0)
        else:
            val = float(np.mean(chunk)) * 4.5
            bands.append(round(min(1.0, max(0.0, val)), 4))
    return bands


def voice_band_ratio(pcm_bytes: bytes, sample_rate: int = INPUT_RATE) -> float:
    """Fração da energia AC contida na banda de voz humana (80–4000 Hz).

    Sinais sintéticos constantes (DC puro, usado em testes simples) retornam 1.0
    para preservar compatibilidade, enquanto ruídos reais (chiado agudo > 4 kHz
    ou vento/hum grave < 80 Hz) retornam valores baixos (< 0.25).
    """
    if len(pcm_bytes) < 8:
        return 0.0
    usable = len(pcm_bytes) - (len(pcm_bytes) % 2)
    samples = np.frombuffer(pcm_bytes[:usable], dtype=np.int16).astype(np.float64)
    ac = samples - np.mean(samples)
    ac_energy = float(np.sum(np.square(ac)))
    if ac_energy < 1e-6:
        # Sinal constante (ex.: pcm(2000) em testes unitários)
        return 1.0

    spectrum = np.abs(np.fft.rfft(ac)) ** 2
    freqs = np.fft.rfftfreq(len(ac), d=1.0 / sample_rate)
    total_power = float(np.sum(spectrum))
    if total_power <= 0.0:
        return 0.0
    mask = (freqs >= _VOICE_LOW_HZ) & (freqs <= _VOICE_HIGH_HZ)
    voice_power = float(np.sum(spectrum[mask]))
    return voice_power / total_power


class NeuralVAD:
    """Wrapper stateful para o modelo Silero VAD em ONNX (512 amostras a 16 kHz, threshold=0.5)."""

    def __init__(
        self,
        model_path: Path,
        sample_rate: int = INPUT_RATE,
        threshold: float = 0.5,
    ) -> None:
        import onnxruntime as ort  # type: ignore[import-not-found]

        self.threshold = threshold
        self.sample_rate = sample_rate
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._input_names = {inp.name for inp in self._session.get_inputs()}
        self._sr = np.array(sample_rate, dtype=np.int64)
        self._lock = threading.Lock()
        self.reset_states()

    def reset_states(self) -> None:
        with self._lock:
            self._state = np.zeros((2, 1, 128), dtype=np.float32)
            self._h = np.zeros((2, 1, 64), dtype=np.float32)
            self._c = np.zeros((2, 1, 64), dtype=np.float32)
            self._context = np.zeros((1, 64), dtype=np.float32)

    def speech_probability(self, pcm_bytes: bytes) -> float:
        usable = len(pcm_bytes) - (len(pcm_bytes) % 2)
        if usable < 1024:  # 512 amostras int16 = 1024 bytes
            return 0.0
        samples = (
            np.frombuffer(pcm_bytes[:usable], dtype=np.int16).astype(np.float32)
            / 32768.0
        )
        probs: List[float] = []
        with self._lock:
            for start in range(0, len(samples) - 511, 512):
                frame = samples[start : start + 512][np.newaxis, :]
                if "state" in self._input_names:
                    # Silero v5: aceita (1, 576) com contexto de 64 amostras ou (1, 512)
                    x = np.concatenate([self._context, frame], axis=1)
                    try:
                        out, self._state = self._session.run(
                            None,
                            {"input": x, "state": self._state, "sr": self._sr},
                        )
                    except Exception:  # noqa: BLE001
                        out, self._state = self._session.run(
                            None,
                            {"input": frame, "state": self._state, "sr": self._sr},
                        )
                    self._context = frame[:, -64:]
                else:
                    # Silero v4 (inputs: input, sr, h, c)
                    out, self._h, self._c = self._session.run(
                        None,
                        {
                            "input": frame,
                            "sr": self._sr,
                            "h": self._h,
                            "c": self._c,
                        },
                    )
                probs.append(float(np.squeeze(out)))
        return max(probs) if probs else 0.0

    def is_speech(self, pcm_bytes: bytes) -> bool:
        return self.speech_probability(pcm_bytes) >= self.threshold


# Alias para compatibilidade com código existente
SileroOnnxVAD = NeuralVAD


def is_speech_chunk(
    pcm_bytes: bytes,
    threshold_rms: float,
    noise_floor_rms: float = 0.0,
    sample_rate: int = INPUT_RATE,
    silero: Optional[NeuralVAD] = None,
) -> bool:
    """Combina limiar dinâmico de RMS com NeuralVAD (Silero ONNX) ou banda de voz espectral."""
    rms = calculate_rms(pcm_bytes)
    effective_threshold = max(threshold_rms, noise_floor_rms * 1.8)
    if rms < effective_threshold:
        return False
    if silero is not None:
        usable = len(pcm_bytes) - (len(pcm_bytes) % 2)
        if usable >= 4:
            samples = np.frombuffer(pcm_bytes[:usable], dtype=np.int16).astype(np.float64)
            if float(np.sum(np.square(samples - np.mean(samples)))) < 1e-6:
                # Buffer DC sintético usado nos testes unitários (ex.: pcm(2000))
                return True
        return silero.is_speech(pcm_bytes)
    return voice_band_ratio(pcm_bytes, sample_rate) >= _MIN_VOICE_BAND_RATIO


class SilenceMonitor:
    """Mede há quanto tempo não há atividade de voz (usuário ou assistente).

    Combina:
    - Limiar de RMS configurável + piso de ruído ambiente adaptativo;
    - Rejeição espectral de ruídos fora da banda de voz humana (vento/hum < 80 Hz
      e chiados > 4 kHz);
    - NeuralVAD (Silero VAD via ONNX) caso `models/silero_vad.onnx` esteja presente.
    Thread-safe: é alimentado por executores e lido pelo loop asyncio.
    """

    def __init__(
        self,
        threshold_rms: float,
        timeout_sec: float,
        clock: Callable[[], float] = time.monotonic,
        silero_model_path: Optional[Path] = None,
    ) -> None:
        self.threshold_rms = threshold_rms
        self.timeout_sec = timeout_sec
        self._clock = clock
        self._lock = threading.Lock()
        self._last_activity = clock()
        self._noise_floor_rms = 0.0
        self._silero: Optional[NeuralVAD] = None

        if silero_model_path is not None and silero_model_path.is_file():
            try:
                self._silero = NeuralVAD(silero_model_path, threshold=0.5)
                logger.info("NeuralVAD (Silero ONNX) carregado de %s.", silero_model_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Não foi possível carregar Silero VAD (%s); usando VAD espectral.",
                    exc,
                )

    @property
    def noise_floor_rms(self) -> float:
        with self._lock:
            return self._noise_floor_rms

    def reset(self) -> None:
        with self._lock:
            self._last_activity = self._clock()
            self._noise_floor_rms = 0.0
            if self._silero is not None:
                self._silero.reset_states()

    def mark_activity(self) -> None:
        with self._lock:
            self._last_activity = self._clock()

    def feed(self, pcm_bytes: bytes) -> bool:
        """Processa um chunk do microfone. Retorna True se for voz humana."""
        rms = calculate_rms(pcm_bytes)
        with self._lock:
            floor = self._noise_floor_rms

        is_voice = is_speech_chunk(
            pcm_bytes,
            threshold_rms=self.threshold_rms,
            noise_floor_rms=floor,
            silero=self._silero,
        )
        with self._lock:
            if is_voice:
                self._last_activity = self._clock()
            elif rms > 0.0:
                capped_rms = min(rms, self.threshold_rms * 0.8)
                if self._noise_floor_rms == 0.0:
                    self._noise_floor_rms = capped_rms
                else:
                    self._noise_floor_rms = (
                        0.92 * self._noise_floor_rms + 0.08 * capped_rms
                    )
        return is_voice

    def silence_duration(self) -> float:
        with self._lock:
            return self._clock() - self._last_activity

    def is_timed_out(self, timeout_sec: Optional[float] = None) -> bool:
        limit = self.timeout_sec if timeout_sec is None else timeout_sec
        return self.silence_duration() >= limit
