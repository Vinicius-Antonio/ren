"""Roteador Determinístico Edge-to-Cloud e Fila de Workers Assíncronos (Slow Loop)."""

from __future__ import annotations

import logging
import re
import threading
import unicodedata
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from src.gateway.ws_server import GatewayServer
from src.system.pc_actions import control_media, open_application

logger = logging.getLogger(__name__)


def _norm(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    no_accents = "".join(c for c in decomposed if not unicodedata.combining(c))
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", no_accents)
    return re.sub(r"\s+", " ", cleaned).strip()


@dataclass(frozen=True)
class EdgeRouteMatch:
    matched: bool
    intent: str = ""
    confirmation_text: str = ""
    widget_type: str = ""
    widget_payload: Optional[Dict[str, Any]] = None


class EdgeRouter:
    """Classificador determinístico local (Edge-to-Cloud) para comandos de SO de latência zero."""

    _VOL_UP_RE = re.compile(
        r"\b(?:aumenta(?:r)?(?: o)? volume|sobe(?: o)? volume|volume mais alto)\b"
    )
    _VOL_DOWN_RE = re.compile(
        r"\b(?:abaixa(?:r)?(?: o)? volume|diminui(?:r)?(?: o)? volume|volume mais baixo)\b"
    )
    _MUTE_RE = re.compile(
        r"\b(?:muta(?:r)?(?: o)? som|silencia(?:r)?(?: o)? pc|tirar(?: o)? som)\b"
    )
    _PAUSE_RE = re.compile(
        r"\b(?:pausa(?:r)?(?: a)? musica|continua(?:r)?(?: a)? musica|play pause)\b"
    )
    _NEXT_RE = re.compile(r"\b(?:proxima musica|pula(?:r)?(?: a)? musica|proxima faixa)\b")
    _OPEN_APP_RE = re.compile(
        r"\b(?:abre|abrir|abra)(?: o| a)? (calculadora|bloco de notas|notepad|explorador de arquivos|terminal|vscode|vs code|spotify|configuracoes)\b"
    )

    def __init__(self, gateway: Optional[GatewayServer] = None) -> None:
        self._gateway = gateway
        self._dispatched_in_turn: set[str] = set()

    def reset_turn(self) -> None:
        self._dispatched_in_turn.clear()

    def try_dispatch(self, transcript: str) -> EdgeRouteMatch:
        """Avalia a transcrição parcial/final e executa comandos diretos do SO sem esperar a nuvem."""
        norm = _norm(transcript)
        if not norm:
            return EdgeRouteMatch(matched=False)

        if self._VOL_UP_RE.search(norm) and "vol_up" not in self._dispatched_in_turn:
            self._dispatched_in_turn.add("vol_up")
            msg = control_media("volume_up", steps=5)
            payload = {"action": "volume_up", "message": msg}
            if self._gateway:
                self._gateway.emit_ui_widget("system_control", payload)
            return EdgeRouteMatch(True, "volume_up", msg, "system_control", payload)

        if self._VOL_DOWN_RE.search(norm) and "vol_down" not in self._dispatched_in_turn:
            self._dispatched_in_turn.add("vol_down")
            msg = control_media("volume_down", steps=5)
            payload = {"action": "volume_down", "message": msg}
            if self._gateway:
                self._gateway.emit_ui_widget("system_control", payload)
            return EdgeRouteMatch(True, "volume_down", msg, "system_control", payload)

        if self._MUTE_RE.search(norm) and "mute" not in self._dispatched_in_turn:
            self._dispatched_in_turn.add("mute")
            msg = control_media("mute")
            payload = {"action": "mute", "message": msg}
            if self._gateway:
                self._gateway.emit_ui_widget("system_control", payload)
            return EdgeRouteMatch(True, "mute", msg, "system_control", payload)

        if self._PAUSE_RE.search(norm) and "play_pause" not in self._dispatched_in_turn:
            self._dispatched_in_turn.add("play_pause")
            msg = control_media("play_pause")
            payload = {"action": "play_pause", "message": msg}
            if self._gateway:
                self._gateway.emit_ui_widget("media_control", payload)
            return EdgeRouteMatch(True, "play_pause", msg, "media_control", payload)

        if self._NEXT_RE.search(norm) and "next" not in self._dispatched_in_turn:
            self._dispatched_in_turn.add("next")
            msg = control_media("next")
            payload = {"action": "next", "message": msg}
            if self._gateway:
                self._gateway.emit_ui_widget("media_control", payload)
            return EdgeRouteMatch(True, "next", msg, "media_control", payload)

        m_app = self._OPEN_APP_RE.search(norm)
        if m_app:
            app_name = m_app.group(1)
            key = f"open:{app_name}"
            if key not in self._dispatched_in_turn:
                self._dispatched_in_turn.add(key)
                msg = open_application(app_name)
                payload = {"app": app_name, "message": msg}
                if self._gateway:
                    self._gateway.emit_ui_widget("app_launcher", payload)
                return EdgeRouteMatch(True, key, msg, "app_launcher", payload)

        return EdgeRouteMatch(matched=False)


class SlowLoopManager:
    """Fila de workers assíncronos em background para tarefas longas (> 5s)."""

    def __init__(
        self,
        gateway: Optional[GatewayServer] = None,
        on_complete_callback: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        self._gateway = gateway
        self._on_complete = on_complete_callback
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="hermes-slow-loop")
        self._lock = threading.Lock()
        self._next_id = 1
        self._tasks: Dict[int, Future[str]] = {}

    def submit_task(
        self,
        task_name: str,
        fn: Callable[..., str],
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Enfileira uma operação demorada no Slow Loop e retorna confirmação imediata para o Fast Loop."""
        with self._lock:
            task_id = self._next_id
            self._next_id += 1

        if self._gateway is not None:
            self._gateway.emit_ui_widget(
                "background_task",
                {
                    "task_id": task_id,
                    "task_name": task_name,
                    "status": "RUNNING",
                },
            )

        def _runner() -> str:
            try:
                result = fn(*args, **kwargs)
                status = "COMPLETED"
            except Exception as exc:  # noqa: BLE001
                logger.warning("Falha na tarefa Slow Loop #%d (%s): %s", task_id, task_name, exc)
                result = f"Falha ao concluir {task_name}: {exc}"
                status = "ERROR"

            if self._gateway is not None:
                self._gateway.emit_ui_widget(
                    "background_task",
                    {
                        "task_id": task_id,
                        "task_name": task_name,
                        "status": status,
                        "result": result,
                    },
                )
            if self._on_complete is not None:
                try:
                    self._on_complete(task_name, result)
                except Exception:  # noqa: BLE001
                    logger.debug("Erro no callback do Slow Loop", exc_info=True)
            return result

        fut = self._pool.submit(_runner)
        with self._lock:
            self._tasks[task_id] = fut

        return {
            "task_id": task_id,
            "status": "STARTED",
            "consent_message": (
                f"Tarefa '{task_name}' iniciada em segundo plano (ID #{task_id}). "
                "Avise o usuário imediatamente que você já está processando e notificará assim que terminar."
            ),
        }

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False)

