"""Bips curtos de ativação/desativação sintetizados com NumPy."""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from src.config import OUTPUT_RATE

logger = logging.getLogger(__name__)

_AMPLITUDE = 0.25  # fração do fundo de escala int16 — discreto
_FADE_SEC = 0.008  # evita "clicks" nas bordas


def _tone(freq_hz: float, duration_sec: float, rate: int) -> np.ndarray:
    n = int(rate * duration_sec)
    t = np.arange(n, dtype=np.float64) / rate
    wave = np.sin(2 * np.pi * freq_hz * t)
    fade = min(int(rate * _FADE_SEC), n // 2)
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade)
        wave[:fade] *= ramp
        wave[-fade:] *= ramp[::-1]
    return wave


def synth_beep(
    freqs_hz: Sequence[float], tone_sec: float = 0.07, rate: int = OUTPUT_RATE
) -> bytes:
    """Concatena tons e retorna PCM 16-bit mono."""
    if not freqs_hz:
        return b""
    wave = np.concatenate([_tone(f, tone_sec, rate) for f in freqs_hz])
    pcm = np.clip(wave * _AMPLITUDE * 32767, -32768, 32767).astype(np.int16)
    return pcm.tobytes()


ACTIVATION_BEEP: bytes = synth_beep([660.0, 880.0])
DEACTIVATION_BEEP: bytes = synth_beep([660.0, 440.0])


def play_pcm(stream_manager: object, pcm: bytes, rate: int = OUTPUT_RATE) -> None:
    """Toca PCM de forma síncrona. Erros de áudio não derrubam o Hermes."""
    try:
        with stream_manager.output_stream(rate) as out:  # type: ignore[attr-defined]
            out.write(pcm)
    except Exception:  # noqa: BLE001
        logger.warning("Não foi possível tocar o bip.", exc_info=True)

