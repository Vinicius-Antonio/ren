"""Compactação incremental de contexto a cada 15 turnos não compactados."""

from __future__ import annotations

import inspect
import logging
from typing import Any, Dict, List, Optional, Sequence

from src.memory.sqlite_store import EpisodicMemoryStore

logger = logging.getLogger(__name__)

DEFAULT_COMPACT_THRESHOLD: int = 15
DEFAULT_COMPACT_MODEL: str = "gemini-2.5-flash"

_COMPACT_PROMPT: str = """\
Você é o compactador de contexto de voz do Hermes.
Resuma os turnos recentes abaixo preservando decisões, preferências, nomes, erros discutidos e tarefas pendentes.
Incorpore o resumo anterior (se houver) em um único parágrafo conciso em português do Brasil (máximo de 600 caracteres).
"""


class ContextCompactor:
    """Compacta turnos não compactados no SQLite e injeta `rolling_summary` na sessão Live."""

    def __init__(
        self,
        threshold: int = DEFAULT_COMPACT_THRESHOLD,
        model_name: str = DEFAULT_COMPACT_MODEL,
    ) -> None:
        self.threshold = max(1, threshold)
        self.model_name = model_name

    @staticmethod
    def _fallback_summarize(
        previous_summary: str, turns: Sequence[Dict[str, Any]]
    ) -> str:
        snippets: List[str] = []
        for t in turns[-6:]:
            u = str(t.get("user_text", "") or t.get("user", "")).strip()
            a = str(t.get("assistant_text", "") or t.get("assistant", "")).strip()
            if u and a:
                snippets.append(f"U: {u[:60]} -> H: {a[:70]}")
            elif u:
                snippets.append(f"U: {u[:80]}")
            elif a:
                snippets.append(f"H: {a[:80]}")
        combined = " | ".join(snippets)
        if previous_summary:
            combined = f"{previous_summary} | {combined}".strip(" |")
        return combined[-600:].strip()

    def _summarize_turns(
        self,
        previous_summary: str,
        turns: Sequence[Dict[str, Any]],
        genai_client: Optional[Any],
    ) -> str:
        if genai_client is not None and hasattr(genai_client, "models"):
            try:
                lines = [
                    f"Usuário: {t.get('user_text', '')}\nHermes: {t.get('assistant_text', '')}"
                    for t in turns
                ]
                prompt = (
                    f"{_COMPACT_PROMPT}\n"
                    f"Resumo anterior: {previous_summary or '(nenhum)'}\n\n"
                    f"Turnos novos:\n" + "\n".join(lines)
                )
                resp = genai_client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )
                text = " ".join((getattr(resp, "text", "") or "").split()).strip()
                if text:
                    return text[:800]
            except Exception as exc:  # noqa: BLE001
                logger.debug("Fallback local na compactação de turnos: %s", exc)
        return self._fallback_summarize(previous_summary, turns)

    def compact_if_needed(
        self,
        store: EpisodicMemoryStore,
        genai_client: Optional[Any] = None,
        session_id: Optional[int] = None,
    ) -> Optional[str]:
        """Compacta quando houver >= `threshold` turnos não compactados e retorna o novo `rolling_summary`."""
        uncompacted = store.get_uncompacted_turns(session_id=session_id)
        if len(uncompacted) < self.threshold:
            return None

        previous = store.get_latest_rolling_summary(session_id=session_id)
        new_summary = self._summarize_turns(previous, uncompacted, genai_client)
        store.update_session_rolling_summary(session_id, new_summary)
        store.mark_turns_compacted([int(t["id"]) for t in uncompacted])
        logger.info(
            "Compactados %d turnos no SQLite (rolling_summary=%d chars).",
            len(uncompacted),
            len(new_summary),
        )
        return new_summary

    @staticmethod
    async def inject_rolling_summary(
        live_session: Any, rolling_summary: str
    ) -> bool:
        """Injeta o `rolling_summary` silenciosamente na sessão Live com `turn_complete=False`."""
        if live_session is None or not rolling_summary.strip():
            return False
        send_client = getattr(live_session, "send_client_content", None)
        if send_client is None:
            return False
        from google.genai import types

        res = send_client(
            turns=types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=f"[Contexto Compactado da Sessão]: {rolling_summary.strip()}"
                    )
                ],
            ),
            turn_complete=False,
        )
        if inspect.isawaitable(res):
            await res
        return True

    async def maybe_compact_and_inject(
        self,
        store: EpisodicMemoryStore,
        genai_client: Optional[Any] = None,
        live_session: Optional[Any] = None,
        session_id: Optional[int] = None,
    ) -> Optional[str]:
        """Executa compactação quando aplicável e injeta o resumo na sessão Live."""
        summary = self.compact_if_needed(
            store=store,
            genai_client=genai_client,
            session_id=session_id,
        )
        if summary and live_session is not None:
            await self.inject_rolling_summary(live_session, summary)
        return summary

