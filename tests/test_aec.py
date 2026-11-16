from __future__ import annotations

import numpy as np
import pytest

from src.audio.aec import AcousticEchoCanceller, resample_24k_to_16k
from src.audio.vad import calculate_rms
from src.config import CHUNK_SIZE, INPUT_RATE, OUTPUT_RATE


def _sine_pcm(freq_hz: float, amplitude: float, rate: int, n_samples: int) -> bytes:
    t = np.arange(n_samples, dtype=np.float64) / rate
    wave = (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.int16)
    return wave.tobytes()


def test_resample_24k_to_16k_length_and_amplitude() -> None:
    # 1536 amostras a 24 kHz equivalem exatamente a 1024 amostras a 16 kHz (64 ms)
    pcm_24k = _sine_pcm(440.0, 8000.0, OUTPUT_RATE, 1536)
    resampled = resample_24k_to_16k(pcm_24k)
    assert len(resampled) == CHUNK_SIZE
    rms = float(np.sqrt(np.mean(np.square(resampled))))
    assert rms == pytest.approx(8000.0 / np.sqrt(2), rel=0.05)
    assert resample_24k_to_16k(b"").size == 0


def test_aec_suppresses_echo_from_speaker() -> None:
    aec = AcousticEchoCanceller(frame_size=CHUNK_SIZE, step_size=0.4)
    ref_24k = _sine_pcm(500.0, 6000.0, OUTPUT_RATE, 1536)
    # Simula o eco chegando ao microfone em 16 kHz com atenuação acústica de 50%
    echo_16k = _sine_pcm(500.0, 3000.0, INPUT_RATE, CHUNK_SIZE)

    initial_rms = calculate_rms(echo_16k)
    cleaned_rms = initial_rms
    for _ in range(12):
        aec.feed_reference(ref_24k)
        cleaned_pcm, _ = aec.process(echo_16k, assistant_speaking=True)
        cleaned_rms = calculate_rms(cleaned_pcm)

    # Após convergir, o eco residual deve ser significativamente menor que o original
    assert cleaned_rms < initial_rms * 0.5


def test_aec_detects_barge_in_when_user_speaks_over_assistant() -> None:
    aec = AcousticEchoCanceller(
        frame_size=CHUNK_SIZE, barge_in_rms_threshold=700.0, barge_in_frames=2
    )
    # Voz forte na faixa humana (300 Hz, amplitude 4000 -> RMS ~2828)
    user_voice_16k = _sine_pcm(300.0, 4000.0, INPUT_RATE, CHUNK_SIZE)

    _, barge_1 = aec.process(user_voice_16k, assistant_speaking=True)
    assert not barge_1  # 1º quadro ainda aguarda confirmação
    _, barge_2 = aec.process(user_voice_16k, assistant_speaking=True)
    assert barge_2  # 2º quadro consecutivo confirma barge-in!

