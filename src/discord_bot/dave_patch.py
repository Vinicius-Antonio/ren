"""Patch de compatibilidade entre discord.py 2.7.x e davey >= 0.2.0 (DAVE E2EE).

O discord.py 2.7.1 foi escrito para o `davey` 0.1.x, onde processar o MLS Welcome/Commit
já ativava as novas chaves. A partir do `davey` 0.2.0, as chaves só passam a valer depois de
`DaveSession.prepare_transition()` + `DaveSession.execute_transition()`, que o discord.py nunca chama.

Resultado sem o patch: `dave_session.ready` fica `False` para sempre, o discord.py envia o Opus
SEM a camada DAVE e os clientes da call (que exigem E2EE) descartam todo o áudio do bot.
"""

from __future__ import annotations

import logging
import struct
from typing import Any, Optional

logger = logging.getLogger(__name__)

_PATCHED: bool = False

_MLS_ANNOUNCE_COMMIT_TRANSITION = 29
_MLS_WELCOME = 30


def _session(state: Any) -> Optional[Any]:
    session = getattr(state, "dave_session", None)
    if session is None or not hasattr(session, "prepare_transition"):
        return None
    return session


def _prepare(state: Any, transition_id: int, protocol_version: int) -> None:
    session = _session(state)
    if session is None:
        return
    try:
        session.prepare_transition(int(transition_id), int(protocol_version))
        logger.debug("DAVE prepare_transition(%s, v%s)", transition_id, protocol_version)
    except Exception as exc:  # noqa: BLE001
        logger.debug("DAVE prepare_transition(%s) falhou: %s", transition_id, exc)


def _execute(state: Any, transition_id: int) -> None:
    session = _session(state)
    if session is None:
        return
    try:
        executed = session.execute_transition(int(transition_id))
        logger.debug(
            "DAVE execute_transition(%s) -> %s (ready=%s)",
            transition_id,
            executed,
            getattr(session, "ready", None),
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("DAVE execute_transition(%s) falhou: %s", transition_id, exc)


def apply_dave_compat_patch() -> bool:
    """Aplica (uma única vez) o patch. Retorna True se o patch estiver ativo."""
    global _PATCHED
    if _PATCHED:
        return True
    try:
        import davey  # type: ignore[import-not-found]
        from discord.gateway import DiscordVoiceWebSocket
        from discord.voice_state import VoiceConnectionState
    except Exception:  # noqa: BLE001
        return False

    if not hasattr(davey.DaveSession, "prepare_transition"):
        # davey 0.1.x: o discord.py já funciona sem patch.
        _PATCHED = True
        return True

    orig_send_ready = DiscordVoiceWebSocket.send_transition_ready
    orig_binary = DiscordVoiceWebSocket.received_binary_message
    orig_execute = VoiceConnectionState._execute_transition

    async def send_transition_ready(self: Any, transition_id: int) -> None:
        # Chamado após Welcome/Commit/PrepareTransition, com a transição já registrada como pendente.
        state = self._connection
        version = state.dave_pending_transitions.get(transition_id, state.dave_protocol_version)
        _prepare(state, transition_id, version)
        await orig_send_ready(self, transition_id)

    async def received_binary_message(self: Any, msg: bytes) -> None:
        await orig_binary(self, msg)
        # transition_id == 0 em Welcome/Commit = aplicar imediatamente (sem op 22 do servidor).
        try:
            op = msg[2]
            if op in (_MLS_ANNOUNCE_COMMIT_TRANSITION, _MLS_WELCOME) and len(msg) >= 5:
                transition_id = struct.unpack_from(">H", msg, 3)[0]
                if transition_id == 0:
                    state = self._connection
                    _prepare(state, 0, state.dave_protocol_version)
                    _execute(state, 0)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Falha no pós-processamento DAVE binário: %s", exc)

    async def _execute_transition(self: Any, transition_id: int) -> None:
        version = self.dave_pending_transitions.get(transition_id, self.dave_protocol_version)
        await orig_execute(self, transition_id)
        if transition_id == 0:
            _prepare(self, 0, version)
        _execute(self, transition_id)

    DiscordVoiceWebSocket.send_transition_ready = send_transition_ready  # type: ignore[method-assign]
    DiscordVoiceWebSocket.received_binary_message = received_binary_message  # type: ignore[method-assign]
    VoiceConnectionState._execute_transition = _execute_transition  # type: ignore[method-assign]

    # Sincroniza secret_key do AudioReader quando o Discord rotaciona a chave ou reconecta a voz.
    try:
        from discord.ext.voice_recv import reader as voice_reader  # type: ignore[import-not-found]

        orig_reader_callback = voice_reader.AudioReader.callback

        def synced_reader_callback(self: Any, packet_data: bytes) -> None:
            vc = getattr(self, "voice_client", None)
            raw_key = getattr(vc, "secret_key", None)
            if raw_key:
                current_key = bytes(raw_key)
                if getattr(self, "_hermes_last_key", None) != current_key:
                    self._hermes_last_key = current_key
                    try:
                        self.update_secret_key(current_key)
                        logger.info("Chave de áudio do Discord sincronizada no AudioReader.")
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("Falha ao sincronizar secret_key no AudioReader: %s", exc)
            orig_reader_callback(self, packet_data)

        voice_reader.AudioReader.callback = synced_reader_callback  # type: ignore[method-assign]
    except Exception:  # noqa: BLE001
        pass

    _PATCHED = True
    logger.info("Patch de compatibilidade DAVE (davey >= 0.2) aplicado ao discord.py.")
    return True

