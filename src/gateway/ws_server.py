"""Servidor WebSocket local (`127.0.0.1:8765`) para telemetria em tempo real com o HUD 3D.

Contrato de Mensagens (JSON Outbound para o HUD):
- `STATE_CHANGED`: `{"type": "STATE_CHANGED", "status": "STANDBY" | "LISTENING" | "PROCESSING" | "SPEAKING"}`
- `AUDIO_LEVEL`:   `{"type": "AUDIO_LEVEL", "rms": float, "frequencies": [float]}` (throttled a 50ms)
- `UI_WIDGET`:     `{"type": "UI_WIDGET", "widget": str, "payload": dict}`
"""

from __future__ import annotations

import asyncio
import enum
import json
import logging
import threading
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional, Set

from src.audio.vad import calculate_rms, compute_fft_bands

logger = logging.getLogger(__name__)

DEFAULT_GATEWAY_HOST: str = "127.0.0.1"
DEFAULT_GATEWAY_PORT: int = 8765
_AUDIO_TELEMETRY_INTERVAL_SEC: float = 0.05  # 50 ms (20 Hz)


class GatewayStatus(str, enum.Enum):
    STANDBY = "STANDBY"
    LISTENING = "LISTENING"
    PROCESSING = "PROCESSING"
    SPEAKING = "SPEAKING"


class GatewayServer:
    """Servidor WebSocket local assíncrono com bridge thread-safe para o HUD Tauri/Three.js."""

    def __init__(
        self,
        host: str = DEFAULT_GATEWAY_HOST,
        port: int = DEFAULT_GATEWAY_PORT,
        enabled: bool = True,
    ) -> None:
        self.host = host
        self.port = port
        self.enabled = enabled

        self._lock = threading.Lock()
        self._current_status: GatewayStatus = GatewayStatus.STANDBY
        self._last_audio_emit: float = 0.0
        self._history: Deque[Dict[str, Any]] = deque(maxlen=100)
        self._command_handlers: List[Callable[[Dict[str, Any]], None]] = []

        self._clients: Set[Any] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event: Optional[asyncio.Event] = None

    @property
    def current_status(self) -> GatewayStatus:
        with self._lock:
            return self._current_status

    @property
    def recent_messages(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._history)

    def on_command(self, handler: Callable[[Dict[str, Any]], None]) -> None:
        with self._lock:
            self._command_handlers.append(handler)

    # --------------------------------------------------- ciclo de vida (thread)
    def start_background(self) -> None:
        """Inicia o servidor WebSocket `127.0.0.1:8765` em uma thread daemon dedicada."""
        if not self.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run_thread, name="hermes-gateway-ws", daemon=True
        )
        self._thread.start()

    def _run_thread(self) -> None:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            loop.run_until_complete(self._serve_forever())
        except Exception as exc:  # noqa: BLE001
            logger.debug("Gateway WebSocket encerrado ou indisponível: %s", exc)
        finally:
            self._loop = None

    async def _serve_forever(self) -> None:
        import websockets

        self._stop_event = asyncio.Event()
        try:
            async with websockets.serve(
                self._handle_client, self.host, self.port
            ):
                logger.info(
                    "Gateway WebSocket Hermes 2.0 ativo em ws://%s:%d",
                    self.host,
                    self.port,
                )
                await self._stop_event.wait()
        except OSError as exc:
            logger.warning(
                "Não foi possível abrir porta %d do Gateway WebSocket (%s).",
                self.port,
                exc,
            )

    async def _handle_client(self, websocket: Any) -> None:
        self._clients.add(websocket)
        try:
            # Envia o estado atual imediatamente ao conectar
            init_msg = {
                "type": "STATE_CHANGED",
                "status": self.current_status.value,
            }
            await websocket.send(json.dumps(init_msg, ensure_ascii=False))
            async for raw in websocket:
                try:
                    data = json.loads(raw)
                    if isinstance(data, dict):
                        self._dispatch_inbound(data)
                except Exception:  # noqa: BLE001
                    logger.debug("Mensagem inválida recebida no Gateway: %r", raw)
        except Exception:  # noqa: BLE001
            pass
        finally:
            self._clients.discard(websocket)

    def _dispatch_inbound(self, payload: Dict[str, Any]) -> None:
        with self._lock:
            handlers = list(self._command_handlers)
        for h in handlers:
            try:
                h(payload)
            except Exception:  # noqa: BLE001
                logger.debug("Erro em handler de comando do Gateway", exc_info=True)

    def close(self) -> None:
        loop = self._loop
        stop_ev = self._stop_event
        if loop is not None and stop_ev is not None and loop.is_running():
            loop.call_soon_threadsafe(stop_ev.set)

    # ---------------------------------------------------- emissão de mensagens
    def _broadcast_dict(self, message: Dict[str, Any]) -> None:
        with self._lock:
            self._history.append(message)
        if not self.enabled:
            return
        loop = self._loop
        if loop is None or not loop.is_running() or not self._clients:
            return
        raw = json.dumps(message, ensure_ascii=False)
        loop.call_soon_threadsafe(
            lambda: asyncio.create_task(self._async_broadcast(raw))
        )

    async def _async_broadcast(self, raw_json: str) -> None:
        if not self._clients:
            return
        dead: List[Any] = []
        for ws in list(self._clients):
            try:
                await ws.send(raw_json)
            except Exception:  # noqa: BLE001
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)

    def emit_state_changed(self, status: GatewayStatus | str) -> Dict[str, Any]:
        """Emite `STATE_CHANGED`: `{"type": "STATE_CHANGED", "status": ...}`."""
        resolved = (
            status
            if isinstance(status, GatewayStatus)
            else GatewayStatus(str(status).upper())
        )
        with self._lock:
            self._current_status = resolved
        msg = {
            "type": "STATE_CHANGED",
            "status": resolved.value,
        }
        self._broadcast_dict(msg)
        return msg

    def emit_audio_level(
        self,
        rms: float,
        frequencies: List[float],
        force: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Emite `AUDIO_LEVEL`: `{"type": "AUDIO_LEVEL", "rms": float, "frequencies": [float]}` (a cada 50ms)."""
        now = time.monotonic()
        with self._lock:
            if not force and (now - self._last_audio_emit) < _AUDIO_TELEMETRY_INTERVAL_SEC:
                return None
            self._last_audio_emit = now

        normalized_rms = round(min(1.0, max(0.0, float(rms) / 8000.0)), 4)
        msg = {
            "type": "AUDIO_LEVEL",
            "rms": normalized_rms,
            "raw_rms": round(float(rms), 2),
            "frequencies": [round(float(f), 4) for f in frequencies],
        }
        self._broadcast_dict(msg)
        return msg

    def emit_audio_chunk_telemetry(
        self, pcm_bytes: bytes, sample_rate: int = 16_000, force: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Calcula RMS + 16 bandas FFT de um chunk PCM e emite `AUDIO_LEVEL` se o intervalo de 50ms passou."""
        now = time.monotonic()
        with self._lock:
            if not force and (now - self._last_audio_emit) < _AUDIO_TELEMETRY_INTERVAL_SEC:
                return None
        rms = calculate_rms(pcm_bytes)
        freqs = compute_fft_bands(pcm_bytes, sample_rate=sample_rate, num_bands=16)
        return self.emit_audio_level(rms=rms, frequencies=freqs, force=True)

    def emit_ui_widget(self, widget: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Emite `UI_WIDGET`: `{"type": "UI_WIDGET", "widget": string, "payload": object}`."""
        msg = {
            "type": "UI_WIDGET",
            "widget": str(widget),
            "payload": dict(payload),
        }
        self._broadcast_dict(msg)
        return msg

