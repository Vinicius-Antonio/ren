"""Fakes compartilhados: PyAudio, streams, Porcupine e cliente Gemini Live."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, List, Optional

import numpy as np
import pytest

from src.config import Settings


def make_settings(**overrides: Any) -> Settings:
    base = dict(
        gemini_api_key="test-key",
        gemini_model="test-model",
        gemini_voice="Puck",
        picovoice_access_key=None,
        wake_word_model_path=Path("does/not/exist.ppn"),
        vosk_model_path=Path("does/not/exist_vosk_dir"),
        wake_word_builtin="jarvis",
        wake_word_sensitivity=0.6,
        vad_silence_threshold_rms=500.0,
        vad_silence_timeout_sec=0.3,
        half_duplex=True,
        sound_fx_enabled=False,
    )
    base.update(overrides)
    return Settings(**base)


def pcm(amplitude: int, n: int = 1024) -> bytes:
    return np.full(n, amplitude, dtype=np.int16).tobytes()


class FakeStream:
    def __init__(self, input_data: Optional[bytes] = None, read_delay: float = 0.005):
        self.input_data = input_data if input_data is not None else pcm(0)
        self.read_delay = read_delay
        self.written: List[bytes] = []
        self.active = True
        self.stopped = False
        self.closed = False
        self.fail_read = False

    def read(self, n: int, exception_on_overflow: bool = True) -> bytes:
        if self.closed:
            raise OSError("read em stream fechado")
        if self.fail_read:
            raise OSError("falha de leitura")
        time.sleep(self.read_delay)
        return self.input_data

    def write(self, data: bytes) -> None:
        if self.closed:
            raise OSError("write em stream fechado")
        self.written.append(data)

    def is_active(self) -> bool:
        return self.active

    def stop_stream(self) -> None:
        self.stopped = True
        self.active = False

    def close(self) -> None:
        self.closed = True


class FakePyAudioInstance:
    def __init__(self, module: "FakePyAudioModule") -> None:
        self.module = module
        self.terminated = False

    def open(self, **kwargs: Any) -> FakeStream:
        self.module.open_calls.append(kwargs)
        if self.module.fail_open:
            raise OSError("dispositivo indisponível")
        stream = FakeStream(self.module.input_data)
        self.module.streams.append(stream)
        return stream

    def terminate(self) -> None:
        self.terminated = True


class FakePyAudioModule:
    paInt16 = 8

    def __init__(self, input_data: Optional[bytes] = None) -> None:
        self.open_calls: List[dict] = []
        self.streams: List[FakeStream] = []
        self.instances: List[FakePyAudioInstance] = []
        self.input_data = input_data
        self.fail_open = False

    def PyAudio(self) -> FakePyAudioInstance:  # noqa: N802 - espelha a API real
        inst = FakePyAudioInstance(self)
        self.instances.append(inst)
        return inst


# ----------------------------------------------------------------- Gemini fakes
def audio_message(data: bytes, turn_complete: bool = False) -> SimpleNamespace:
    part = SimpleNamespace(inline_data=SimpleNamespace(data=data))
    return SimpleNamespace(
        go_away=None,
        server_content=SimpleNamespace(
            model_turn=SimpleNamespace(parts=[part]),
            turn_complete=turn_complete,
            interrupted=False,
            input_transcription=None,
        ),
    )


def transcription_message(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        go_away=None,
        server_content=SimpleNamespace(
            model_turn=None,
            turn_complete=False,
            interrupted=False,
            input_transcription=SimpleNamespace(text=text),
        ),
    )


def turn_complete_message() -> SimpleNamespace:
    return SimpleNamespace(
        go_away=None,
        server_content=SimpleNamespace(
            model_turn=None, turn_complete=True, interrupted=False, input_transcription=None
        ),
    )


def interrupted_message() -> SimpleNamespace:
    return SimpleNamespace(
        go_away=None,
        server_content=SimpleNamespace(
            model_turn=None, turn_complete=False, interrupted=True, input_transcription=None
        ),
    )


class FakeLiveSession:
    def __init__(
        self, turns: Iterable[List[Any]] = (), receive_error: Optional[Exception] = None
    ) -> None:
        self.turns = [list(t) for t in turns]
        self.receive_error = receive_error

    def receive(self) -> Any:
        return self._gen()

    async def _gen(self) -> Any:
        if self.receive_error is not None:
            raise self.receive_error
        if self.turns:
            for msg in self.turns.pop(0):
                yield msg
        else:
            await asyncio.sleep(3600)  # conexão ociosa


class _ConnectCM:
    def __init__(self, client: "FakeGenaiClient") -> None:
        self.client = client

    async def __aenter__(self) -> FakeLiveSession:
        if self.client.connect_error is not None:
            raise self.client.connect_error
        self.client.connected = True
        return self.client.live_session

    async def __aexit__(self, *exc: object) -> None:
        self.client.disconnected = True


class FakeGenaiClient:
    def __init__(
        self,
        live_session: Optional[FakeLiveSession] = None,
        connect_error: Optional[Exception] = None,
    ) -> None:
        self.live_session = live_session or FakeLiveSession()
        self.connect_error = connect_error
        self.connected = False
        self.disconnected = False
        self.connect_kwargs: dict = {}
        self.aio = SimpleNamespace(live=SimpleNamespace(connect=self._connect))

    def _connect(self, **kwargs: Any) -> _ConnectCM:
        self.connect_kwargs = kwargs
        return _ConnectCM(self)


@pytest.fixture
def fake_pyaudio() -> FakePyAudioModule:
    return FakePyAudioModule()

