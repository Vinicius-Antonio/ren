"""Abertura segura e liberação de streams PyAudio."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import Executor
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from src.config import CHANNELS, CHUNK_SIZE, INPUT_RATE, OUTPUT_RATE

logger = logging.getLogger(__name__)


def _load_pyaudio() -> Any:
    import pyaudio  # import tardio: permite testes sem PortAudio instalado

    return pyaudio


def close_stream(stream: Optional[Any]) -> None:
    """Para e fecha um stream sem propagar erros (uso em `finally`)."""
    if stream is None:
        return
    try:
        if stream.is_active():
            stream.stop_stream()
    except Exception:  # noqa: BLE001 - liberação best-effort
        logger.debug("Falha ao parar stream", exc_info=True)
    try:
        stream.close()
    except Exception:  # noqa: BLE001
        logger.debug("Falha ao fechar stream", exc_info=True)


class AudioStreamManager:
    """Dono de uma instância PyAudio; cria streams de 16 kHz (in) e 24 kHz (out)."""

    def __init__(self, pyaudio_module: Optional[Any] = None) -> None:
        self._pa_mod = pyaudio_module or _load_pyaudio()
        self._pa: Optional[Any] = self._pa_mod.PyAudio()

    @property
    def pa(self) -> Any:
        if self._pa is None:
            raise RuntimeError("AudioStreamManager já foi encerrado.")
        return self._pa

    def open_input(
        self, rate: int = INPUT_RATE, frames_per_buffer: int = CHUNK_SIZE
    ) -> Any:
        return self.pa.open(
            format=self._pa_mod.paInt16,
            channels=CHANNELS,
            rate=rate,
            input=True,
            frames_per_buffer=frames_per_buffer,
        )

    def open_output(self, rate: int = OUTPUT_RATE) -> Any:
        return self.pa.open(
            format=self._pa_mod.paInt16,
            channels=CHANNELS,
            rate=rate,
            output=True,
        )

    @contextmanager
    def input_stream(
        self, rate: int = INPUT_RATE, frames_per_buffer: int = CHUNK_SIZE
    ) -> Iterator[Any]:
        stream = None
        try:
            stream = self.open_input(rate, frames_per_buffer)
            yield stream
        finally:
            close_stream(stream)

    @contextmanager
    def output_stream(self, rate: int = OUTPUT_RATE) -> Iterator[Any]:
        stream = None
        try:
            stream = self.open_output(rate)
            yield stream
        finally:
            close_stream(stream)

    def terminate(self) -> None:
        if self._pa is not None:
            try:
                self._pa.terminate()
            finally:
                self._pa = None

    def __enter__(self) -> "AudioStreamManager":
        return self

    def __exit__(self, *exc: object) -> None:
        self.terminate()


async def read_chunk(
    stream: Any, chunk_size: int = CHUNK_SIZE, executor: Optional[Executor] = None
) -> bytes:
    """`stream.read` bloqueante executado fora do event loop."""
    loop = asyncio.get_running_loop()
    data: bytes = await loop.run_in_executor(executor, stream.read, chunk_size, False)
    return data


async def write_chunk(
    stream: Any, data: bytes, executor: Optional[Executor] = None
) -> None:
    """`stream.write` bloqueante executado fora do event loop."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(executor, stream.write, data)
