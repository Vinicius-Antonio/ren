from __future__ import annotations

import numpy as np
import pytest

from src.audio.sound_fx import ACTIVATION_BEEP, DEACTIVATION_BEEP, synth_beep
from src.audio.vad import SilenceMonitor, calculate_rms, voice_band_ratio
from tests.conftest import pcm


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_rms_silence_is_zero() -> None:
    assert calculate_rms(pcm(0)) == 0.0


def test_rms_constant_signal() -> None:
    assert calculate_rms(pcm(1000)) == pytest.approx(1000.0)


def test_rms_sine_wave() -> None:
    t = np.arange(16000) / 16000
    wave = (10000 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
    assert calculate_rms(wave.tobytes()) == pytest.approx(10000 / np.sqrt(2), rel=1e-3)


def test_rms_empty_and_odd_buffers() -> None:
    assert calculate_rms(b"") == 0.0
    assert calculate_rms(b"\x01") == 0.0
    assert calculate_rms(pcm(1000, 4) + b"\x07") == pytest.approx(1000.0)


def test_rms_no_int16_overflow() -> None:
    assert calculate_rms(pcm(-32768)) == pytest.approx(32768.0)


def test_monitor_times_out_after_silence() -> None:
    clock = FakeClock()
    mon = SilenceMonitor(threshold_rms=500, timeout_sec=6.0, clock=clock)
    clock.now = 5.9
    mon.feed(pcm(10))  # ruído baixo não conta como voz
    assert not mon.is_timed_out()
    clock.now = 6.0
    assert mon.is_timed_out()


def test_monitor_voice_resets_timer() -> None:
    clock = FakeClock()
    mon = SilenceMonitor(threshold_rms=500, timeout_sec=6.0, clock=clock)
    clock.now = 5.0
    assert mon.feed(pcm(2000)) is True
    clock.now = 10.0
    assert not mon.is_timed_out()
    assert mon.silence_duration() == pytest.approx(5.0)
    clock.now = 11.0
    assert mon.is_timed_out()


def test_monitor_assistant_activity_resets_timer() -> None:
    clock = FakeClock()
    mon = SilenceMonitor(threshold_rms=500, timeout_sec=6.0, clock=clock)
    clock.now = 5.5
    mon.mark_activity()  # Hermes tocando áudio
    clock.now = 11.0
    assert not mon.is_timed_out()


def test_beeps_are_valid_pcm() -> None:
    for beep in (ACTIVATION_BEEP, DEACTIVATION_BEEP):
        assert len(beep) > 0 and len(beep) % 2 == 0
        samples = np.frombuffer(beep, dtype=np.int16)
        assert np.abs(samples).max() < 32767 * 0.3
        assert abs(int(samples[0])) < 100  # fade-in evita click
    assert synth_beep([]) == b""


def test_spectral_vad_rejects_high_frequency_hiss_and_accepts_speech_band() -> None:
    clock = FakeClock()
    mon = SilenceMonitor(threshold_rms=500, timeout_sec=6.0, clock=clock)
    t = np.arange(1024) / 16000.0
    # Chiado agudo em 6500 Hz (fora da faixa de voz 80-4000 Hz) com RMS alto (~1414)
    hiss = (2000 * np.sin(2 * np.pi * 6500 * t)).astype(np.int16).tobytes()
    assert voice_band_ratio(hiss) < 0.05
    assert mon.feed(hiss) is False

    # Voz simulada em 300 Hz (dentro da faixa de voz) com mesma amplitude
    speech = (2000 * np.sin(2 * np.pi * 300 * t)).astype(np.int16).tobytes()
    assert voice_band_ratio(speech) > 0.9
    assert mon.feed(speech) is True


