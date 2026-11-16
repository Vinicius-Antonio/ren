from __future__ import annotations

import asyncio

import pytest

from src.audio.stream_manager import AudioStreamManager, close_stream, read_chunk, write_chunk
from src.config import CHUNK_SIZE, INPUT_RATE, OUTPUT_RATE
from tests.conftest import FakePyAudioModule, FakeStream


def test_open_input_and_output_params(fake_pyaudio: FakePyAudioModule) -> None:
    mgr = AudioStreamManager(fake_pyaudio)
    mgr.open_input()
    mgr.open_output()
    inp, out = fake_pyaudio.open_calls
    assert inp == dict(
        format=8, channels=1, rate=INPUT_RATE, input=True, frames_per_buffer=CHUNK_SIZE
    )
    assert out == dict(format=8, channels=1, rate=OUTPUT_RATE, output=True)


def test_context_manager_closes_on_exception(fake_pyaudio: FakePyAudioModule) -> None:
    mgr = AudioStreamManager(fake_pyaudio)
    with pytest.raises(RuntimeError):
        with mgr.input_stream():
            raise RuntimeError("boom")
    stream = fake_pyaudio.streams[0]
    assert stream.stopped and stream.closed


def test_close_stream_tolerates_errors() -> None:
    class Broken(FakeStream):
        def stop_stream(self) -> None:
            raise OSError("x")

    s = Broken()
    close_stream(s)  # não propaga
    assert s.closed
    close_stream(None)


def test_terminate_is_idempotent(fake_pyaudio: FakePyAudioModule) -> None:
    with AudioStreamManager(fake_pyaudio) as mgr:
        pass
    assert fake_pyaudio.instances[0].terminated
    mgr.terminate()
    with pytest.raises(RuntimeError):
        mgr.open_input()


def test_async_read_write_run_off_loop() -> None:
    stream = FakeStream(input_data=b"\x01\x00" * 4)

    async def go() -> bytes:
        data = await read_chunk(stream, 4)
        await write_chunk(stream, data)
        return data

    assert asyncio.run(go()) == b"\x01\x00" * 4
    assert stream.written == [b"\x01\x00" * 4]

