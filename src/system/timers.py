"""Gerenciador de Timers e Lembretes Sonoros em Background do Ren."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from src.audio.sound_fx import synth_beep

logger = logging.getLogger(__name__)

# Alerta sonoro triplo alegre (Lá5 -> Dó#6 -> Mi6)
ALARM_BEEP_PCM: bytes = synth_beep([880.0, 1108.7, 1318.5, 880.0, 1318.5], tone_sec=0.09)


@dataclass
class ActiveTimer:
    timer_id: int
    label: str
    end_monotonic: float
    thread: threading.Timer


class TimerManager:
    """Agenda timers em background que tocam alerta sonoro quando o tempo acaba."""

    def __init__(
        self,
        on_fire_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._lock = threading.Lock()
        self._next_id = 1
        self._timers: Dict[int, ActiveTimer] = {}
        self._on_fire = on_fire_callback or self._default_alarm
        self._voice_reminder_cb: Optional[Callable[[str], None]] = None

    def set_voice_reminder_callback(
        self, callback: Optional[Callable[[str], None]]
    ) -> None:
        """Registra callback para o Ren acordar e falar o lembrete em voz alta quando o timer zerar."""
        with self._lock:
            self._voice_reminder_cb = callback

    def _default_alarm(self, label: str) -> None:
        print(f"\n[Ren] ALARME / TIMER FINALIZADO: {label}!", flush=True)
        try:
            from src.audio.sound_fx import play_pcm
            from src.audio.stream_manager import AudioStreamManager

            with AudioStreamManager() as mgr:
                play_pcm(mgr, ALARM_BEEP_PCM)
        except Exception:  # noqa: BLE001
            logger.debug("Não foi possível tocar som do alarme", exc_info=True)
        with self._lock:
            cb = self._voice_reminder_cb
        if cb is not None:
            try:
                cb(label)
            except Exception:  # noqa: BLE001
                logger.debug("Falha ao disparar lembrete falado do timer", exc_info=True)

    def set_timer(self, duration_seconds: int, label: str = "Timer") -> str:
        secs = max(1, int(duration_seconds))
        clean_label = label.strip() or "Timer"

        with self._lock:
            tid = self._next_id
            self._next_id += 1

            def _trigger() -> None:
                with self._lock:
                    self._timers.pop(tid, None)
                self._on_fire(clean_label)

            t = threading.Timer(secs, _trigger)
            t.daemon = True
            self._timers[tid] = ActiveTimer(
                timer_id=tid,
                label=clean_label,
                end_monotonic=time.monotonic() + secs,
                thread=t,
            )
            t.start()

        mins, rem_secs = divmod(secs, 60)
        if mins > 0 and rem_secs > 0:
            human_time = f"{mins} minuto(s) e {rem_secs} segundo(s)"
        elif mins > 0:
            human_time = f"{mins} minuto(s)"
        else:
            human_time = f"{rem_secs} segundo(s)"
        return f"Timer #{tid} ('{clean_label}') programado para daqui a {human_time}."

    def list_timers(self) -> str:
        now = time.monotonic()
        with self._lock:
            if not self._timers:
                return "Nenhum timer ativo no momento."
            items: List[str] = []
            for tid, item in sorted(self._timers.items()):
                rem = max(0, int(round(item.end_monotonic - now)))
                mins, secs = divmod(rem, 60)
                items.append(f"#{tid} '{item.label}': faltam {mins}m{secs:02d}s")
            return "Timers ativos: " + " | ".join(items)

    def cancel_all(self) -> str:
        with self._lock:
            count = len(self._timers)
            for item in self._timers.values():
                item.thread.cancel()
            self._timers.clear()
        return f"{count} timer(s) cancelado(s)."

