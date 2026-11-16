"""Entry point do Hermes: máquina de estados Standby <-> Ativo."""

from __future__ import annotations

import asyncio
import argparse
from dataclasses import replace
import enum
import logging
import sys
from typing import Callable, Optional

from src.audio.sound_fx import ACTIVATION_BEEP, DEACTIVATION_BEEP, play_pcm
from src.audio.stream_manager import AudioStreamManager
from src.config import Settings
from src.detector.wake_word import CallDisconnectedError, WakeWordDetector
from src.discord_bot.bot import DiscordVoiceBot
from src.gateway.ws_server import GatewayServer, GatewayStatus
from src.gemini.session import GeminiLiveSession, SessionEndReason

logger = logging.getLogger("hermes")


class AgentState(enum.Enum):
    STANDBY = "standby"
    ACTIVE = "ativo"


class HermesApp:
    """Orquestra o ciclo: espera wake word -> sessão Gemini -> espera de novo."""

    def __init__(
        self,
        settings: Settings,
        stream_manager: AudioStreamManager,
        detector: WakeWordDetector,
        session: GeminiLiveSession,
        printer: Callable[[str], None] = lambda msg: print(msg, flush=True),
        gateway: Optional[GatewayServer] = None,
        call_only: bool = False,
        discord_bot: Optional[DiscordVoiceBot] = None,
    ) -> None:
        self.settings = settings
        self.streams = stream_manager
        self.detector = detector
        self.session = session
        self.state = AgentState.STANDBY
        self._print = printer
        self.gateway = gateway or GatewayServer(enabled=False)
        self.call_only = call_only
        self.discord_bot = discord_bot
        if hasattr(self.session, "set_wake_callback") and hasattr(
            self.detector, "trigger_wakeup"
        ):
            self.session.set_wake_callback(self.detector.trigger_wakeup)

    def _beep(self, pcm: bytes) -> None:
        if self.settings.sound_fx_enabled and not self.call_only:
            play_pcm(self.streams, pcm)

    def run_once(self) -> SessionEndReason:
        """Um ciclo completo Standby -> Ativo -> Standby."""
        self.state = AgentState.STANDBY
        self.gateway.emit_state_changed(GatewayStatus.STANDBY)
        self._print(f"[Ren] Em espera (diga '{self.detector.label}')")
        # Apenas CPU local aqui: nenhuma conexão de rede é aberta.
        self.detector.wait_for_wake_word()
        if self.call_only and hasattr(self.session, "set_call_speaker"):
            self.session.set_call_speaker(getattr(self.detector, "wake_speaker_id", None))
            if hasattr(self.session, "set_call_handoff_checker") and hasattr(self.detector, "has_pending_call_speaker"):
                self.session.set_call_handoff_checker(self.detector.has_pending_call_speaker)

        self.state = AgentState.ACTIVE
        self.gateway.emit_state_changed(GatewayStatus.LISTENING)
        self._print("[Ren] Ativado! Conectando...")
        self._beep(ACTIVATION_BEEP)
        if self.call_only and hasattr(self.detector, "start_call_monitor"):
            self.detector.start_call_monitor()
        try:
            reason = asyncio.run(self.session.run())
        finally:
            if self.call_only and hasattr(self.detector, "stop_call_monitor"):
                self.detector.stop_call_monitor()
            self.state = AgentState.STANDBY
            self.gateway.emit_state_changed(GatewayStatus.STANDBY)
        self._print(f"[Ren] Sessão encerrada ({reason.value}).")
        self._beep(DEACTIVATION_BEEP)
        return reason

    def run_forever(self) -> None:
        while True:
            if self.call_only and self.discord_bot is not None and not self.discord_bot.is_in_call:
                self._print("[Ren] Reconectando à call...")
                if not self.discord_bot.is_running:
                    self.discord_bot.start()
                result = self.discord_bot.join_call()
                if not self.discord_bot.is_in_call:
                    logger.warning("Reconexão do Discord falhou: %s", result)
                    import time

                    time.sleep(3.0)
                    continue
            try:
                reason = self.run_once()
            except CallDisconnectedError:
                continue
            if reason in (SessionEndReason.SHUTDOWN, SessionEndReason.LEFT_CALL):
                if reason is SessionEndReason.LEFT_CALL:
                    if self.discord_bot is not None and self.discord_bot.is_in_call:
                        self.discord_bot.leave_call()
                    self._print("[Ren] Saindo da call por solicitação. Até logo!")
                else:
                    self._print("[Ren] Programa encerrado por comando de voz. Até logo!")
                break


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # SDKs ruidosos
    for noisy in ("httpx", "websockets", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main() -> int:
    _configure_logging()
    parser = argparse.ArgumentParser(description="Ren: assistente local ou bot de voz")
    parser.add_argument("--mode", choices=("pc", "call"), default="pc")
    parser.add_argument("--channel", default=None, help="Canal de voz para --mode call")
    parser.add_argument("--model", default=None, help="Sobrescreve GEMINI_MODEL apenas nesta execução")
    args = parser.parse_args()
    try:
        settings = Settings.from_env()
        if args.model:
            settings = replace(settings, gemini_model=args.model)
        settings.validate()
    except ValueError as exc:
        logger.error("%s", exc)
        return 2
    if args.mode == "call" and not settings.discord_bot_token:
        logger.error("DISCORD_BOT_TOKEN é obrigatório no modo call.")
        return 2

    streams: Optional[AudioStreamManager] = None
    detector: Optional[WakeWordDetector] = None
    gateway: Optional[GatewayServer] = None
    discord_bot: Optional[DiscordVoiceBot] = None
    try:
        gateway = GatewayServer(
            port=settings.gateway_port, enabled=settings.gateway_enabled and args.mode == "pc"
        )
        gateway.start_background()
        if args.mode == "call":
            discord_bot = DiscordVoiceBot(settings, gateway=gateway)
            if not discord_bot.start():
                raise RuntimeError("Bot do Discord não ficou pronto.")
            join_result = discord_bot.join_call(args.channel)
            if not discord_bot.is_in_call:
                raise RuntimeError(join_result)
            print(f"[Ren] {join_result}", flush=True)
        streams = AudioStreamManager()
        detector = WakeWordDetector(
            settings, streams, discord_bot=discord_bot, call_only=args.mode == "call"
        )
        session = GeminiLiveSession(
            settings, streams, gateway=gateway, discord_bot=discord_bot,
            call_only=args.mode == "call",
        )
        HermesApp(
            settings, streams, detector, session, gateway=gateway,
            call_only=args.mode == "call",
            discord_bot=discord_bot,
        ).run_forever()
    except KeyboardInterrupt:
        print("\n[Ren] Encerrando. Até logo!", flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("Erro fatal: %s", exc)
        return 1
    finally:
        if discord_bot is not None:
            discord_bot.stop()
        if gateway is not None:
            gateway.close()
        if detector is not None:
            detector.close()
        if streams is not None:
            streams.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
