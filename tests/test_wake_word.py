from __future__ import annotations

from pathlib import Path
from typing import Any, List

import pytest

from src.audio.stream_manager import AudioStreamManager
from src.detector.wake_word import CallDisconnectedError, WakeWordDetector, WakeWordMode
from tests.conftest import FakePyAudioModule, make_settings


class FakePorcupine:
    sample_rate = 16000
    frame_length = 512

    def __init__(self, detect_after: int) -> None:
        self.detect_after = detect_after
        self.calls = 0
        self.deleted = False

    def process(self, pcm: List[int]) -> int:
        assert len(pcm) == self.frame_length
        self.calls += 1
        return 0 if self.calls >= self.detect_after else -1

    def delete(self) -> None:
        self.deleted = True


class FakePorcupineModule:
    def __init__(self, detect_after: int = 3) -> None:
        self.create_kwargs: dict = {}
        self.instance = FakePorcupine(detect_after)

    def create(self, **kwargs: Any) -> FakePorcupine:
        self.create_kwargs = kwargs
        return self.instance


def test_keyboard_fallback_without_access_key(fake_pyaudio: FakePyAudioModule) -> None:
    pressed: List[str] = []
    det = WakeWordDetector(
        make_settings(picovoice_access_key=None),
        AudioStreamManager(fake_pyaudio),
        input_fn=lambda p: pressed.append(p) or "",
    )
    assert det.mode is WakeWordMode.KEYBOARD
    det.wait_for_wake_word()
    assert pressed == [""]
    assert fake_pyaudio.open_calls == []  # microfone nem é aberto


def test_vosk_pt_mode_without_picovoice_key(tmp_path: Path) -> None:
    vosk_dir = tmp_path / "vosk-model-small-pt-0.3"
    vosk_dir.mkdir()
    fake_pa = FakePyAudioModule(input_data=b"\x00\x00" * 2048)

    class FakeRecognizer:
        def __init__(self) -> None:
            self.calls = 0

        def Reset(self) -> None:  # noqa: N802
            pass

        def AcceptWaveform(self, data: bytes) -> bool:  # noqa: N802
            self.calls += 1
            return False

        def PartialResult(self) -> str:  # noqa: N802
            return '{"partial": "oi rei"}' if self.calls >= 2 else '{"partial": ""}'

    class FakeVoskModule:
        def Model(self, path: str) -> object:  # noqa: N802
            assert path == str(vosk_dir)
            return object()

        def KaldiRecognizer(self, model: object, rate: int, grammar: str) -> FakeRecognizer:  # noqa: N802
            assert rate == 16000
            assert "ren" in grammar
            return FakeRecognizer()

    det = WakeWordDetector(
        make_settings(picovoice_access_key=None, vosk_model_path=vosk_dir),
        AudioStreamManager(fake_pa),
        vosk_module=FakeVoskModule(),
    )
    assert det.mode is WakeWordMode.VOSK_PT
    assert det.label == "Ren"
    det.wait_for_wake_word()
    assert fake_pa.streams[0].stopped and fake_pa.streams[0].closed


def test_builtin_fallback_when_ppn_missing(fake_pyaudio: FakePyAudioModule) -> None:
    pv = FakePorcupineModule()
    det = WakeWordDetector(
        make_settings(picovoice_access_key="pv"), AudioStreamManager(fake_pyaudio), pv
    )
    assert det.mode is WakeWordMode.BUILTIN_KEYWORD
    assert pv.create_kwargs["keywords"] == ["jarvis"]
    assert det.label == "jarvis"


def test_custom_model_when_ppn_exists(tmp_path: Path, fake_pyaudio: FakePyAudioModule) -> None:
    ppn = tmp_path / "ren.ppn"
    ppn.write_bytes(b"x")
    pv = FakePorcupineModule()
    det = WakeWordDetector(
        make_settings(picovoice_access_key="pv", wake_word_model_path=ppn),
        AudioStreamManager(fake_pyaudio),
        pv,
    )
    assert det.mode is WakeWordMode.CUSTOM_MODEL
    assert pv.create_kwargs["keyword_paths"] == [str(ppn)]
    assert "model_path" not in pv.create_kwargs
    assert det.label == "Ren"


def test_custom_model_passes_portuguese_language_model_when_present(
    tmp_path: Path, fake_pyaudio: FakePyAudioModule
) -> None:
    ppn = tmp_path / "ren.ppn"
    ppn.write_bytes(b"x")
    pv_lang = tmp_path / "porcupine_params_pt.pv"
    pv_lang.write_bytes(b"pt-model")
    pv = FakePorcupineModule()
    det = WakeWordDetector(
        make_settings(
            picovoice_access_key="pv",
            wake_word_model_path=ppn,
            wake_word_language_model_path=pv_lang,
        ),
        AudioStreamManager(fake_pyaudio),
        pv,
    )
    assert det.mode is WakeWordMode.CUSTOM_MODEL
    assert pv.create_kwargs["keyword_paths"] == [str(ppn)]
    assert pv.create_kwargs["model_path"] == str(pv_lang)


def test_custom_model_falls_back_to_builtin_if_language_file_missing(
    tmp_path: Path, fake_pyaudio: FakePyAudioModule
) -> None:
    ppn = tmp_path / "ren.ppn"
    ppn.write_bytes(b"x")
    calls: List[dict] = []

    class LangErrorPorcupine(FakePorcupineModule):
        def create(self, **kwargs: Any) -> FakePorcupine:
            calls.append(kwargs)
            if "keyword_paths" in kwargs:
                raise RuntimeError("Invalid language model for PT keyword")
            return super().create(**kwargs)

    pv = LangErrorPorcupine()
    det = WakeWordDetector(
        make_settings(picovoice_access_key="pv", wake_word_model_path=ppn),
        AudioStreamManager(fake_pyaudio),
        pv,
    )
    assert det.mode is WakeWordMode.BUILTIN_KEYWORD
    assert len(calls) == 2
    assert calls[1]["keywords"] == ["jarvis"]


def test_detection_closes_mic_before_returning() -> None:
    fake_pa = FakePyAudioModule(input_data=b"\x00\x00" * 512)
    pv = FakePorcupineModule(detect_after=3)
    det = WakeWordDetector(
        make_settings(picovoice_access_key="pv"), AudioStreamManager(fake_pa), pv
    )
    det.wait_for_wake_word()
    assert pv.instance.calls == 3
    assert fake_pa.open_calls[0]["frames_per_buffer"] == 512
    assert fake_pa.open_calls[0]["rate"] == 16000
    stream = fake_pa.streams[0]
    assert stream.stopped and stream.closed
    det.close()
    assert pv.instance.deleted


def test_mic_closed_even_on_read_error() -> None:
    fake_pa = FakePyAudioModule(input_data=b"\x00\x00" * 512)
    pv = FakePorcupineModule()

    det = WakeWordDetector(
        make_settings(picovoice_access_key="pv"), AudioStreamManager(fake_pa), pv
    )
    original_open = fake_pa.instances[0].open

    def open_failing(**kw: Any) -> Any:
        s = original_open(**kw)
        s.fail_read = True
        return s

    fake_pa.instances[0].open = open_failing  # type: ignore[method-assign]
    with pytest.raises(OSError):
        det.wait_for_wake_word()
    assert fake_pa.streams[0].closed


def test_call_disconnect_does_not_wait_forever_for_wake_word() -> None:
    import threading
    from types import SimpleNamespace

    det = WakeWordDetector.__new__(WakeWordDetector)
    det._call_only = True
    det._discord_bot = SimpleNamespace(is_in_call=False)
    det._external_wakeup = threading.Event()
    det._vosk_rec = SimpleNamespace(Reset=lambda: None)
    pa = FakePyAudioModule()
    det._streams = AudioStreamManager(pa)

    with pytest.raises(CallDisconnectedError):
        det._wait_vosk()
    assert pa.open_calls == []


def test_call_wake_word_identifies_the_member_who_said_ren(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    model_dir = tmp_path / "vosk"
    model_dir.mkdir()

    class Recognizer:
        def __init__(self) -> None:
            self.heard = b""

        def AcceptWaveform(self, raw: bytes) -> bool:  # noqa: N802
            self.heard += raw
            return False

        def PartialResult(self) -> str:  # noqa: N802
            return '{"partial": "ren"}' if b"B" in self.heard else '{"partial": "conversa"}'

        def Reset(self) -> None:  # noqa: N802
            self.heard = b""

    class Vosk:
        def SetLogLevel(self, level: int) -> None:  # noqa: N802
            pass

        def Model(self, path: str) -> object:  # noqa: N802
            return object()

        def KaldiRecognizer(self, model: object, rate: int, grammar: str) -> Recognizer:  # noqa: N802
            return Recognizer()

    class Bot:
        is_in_call = True

        def reset_call_input(self) -> None:
            pass

        def get_wake_chunks(self, size: int) -> dict[int, bytes]:
            return {101: b"A" * size, 202: b"B" * size}

    monkeypatch.setattr("src.detector.wake_word.time.sleep", lambda _: None)
    det = WakeWordDetector(
        make_settings(vosk_model_path=model_dir), AudioStreamManager(FakePyAudioModule()),
        vosk_module=Vosk(), discord_bot=Bot(), call_only=True,
    )
    det.wait_for_wake_word()
    assert det.wake_speaker_id == 202
    class AnotherCaller(Bot):
        def clear_wake_audio(self) -> None:
            pass

        def get_wake_chunks(self, size: int) -> dict[int, bytes]:
            return {303: b"B" * size}

    det._discord_bot = AnotherCaller()
    det.start_call_monitor()
    try:
        import threading
        threading.Event().wait(0.15)
        assert det.has_pending_call_speaker()
    finally:
        det.stop_call_monitor()


def test_call_pending_member_gets_next_turn(tmp_path: Path) -> None:
    import time

    model_dir = tmp_path / "vosk"
    model_dir.mkdir()
    class Vosk:
        def Model(self, path: str) -> object:  # noqa: N802
            return object()
        def KaldiRecognizer(self, model: object, rate: int, grammar: str) -> object:  # noqa: N802
            return object()

    det = WakeWordDetector(
        make_settings(vosk_model_path=model_dir), AudioStreamManager(FakePyAudioModule()),
        vosk_module=Vosk(), call_only=True,
    )
    with det._pending_lock:
        det._pending_call_speakers.append((202, time.monotonic()))
    assert det.has_pending_call_speaker()
    det.wait_for_wake_word()
    assert det.wake_speaker_id == 202
