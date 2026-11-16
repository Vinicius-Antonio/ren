"""Organizador inteligente de cofre Obsidian via modelo de texto Gemini (Structured Outputs)."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.memory.sqlite_store import EpisodicMemoryStore

logger = logging.getLogger(__name__)

DEFAULT_ORGANIZER_MODEL: str = "gemini-2.5-pro"
FALLBACK_ORGANIZER_MODEL: str = "gemini-2.5-flash"

_ORGANIZER_PROMPT: str = """\
Você é um arquiteto especialista em Second Brain, Zettelkasten e Obsidian.
Analise as notas Markdown existentes no cofre do usuário e organize tudo como um cérebro bem conectado:
1. Preserve 100% dos fatos e informações originais das notas (NUNCA apague conteúdo importante do usuário).
2. Conecte notas relacionadas usando `[[wikilinks]]` reais (apenas para notas existentes ou novas notas/MOCs que você está criando neste plano).
3. Quando fizer sentido, crie ou atualize notas índice / MOCs (Maps of Content, ex.: `00 - Mapa Mental.md` ou `MOCs/...`) agrupando projetos, ideias e notas de voz por tema para enriquecer o Graph View.
4. Estruture cada nota com títulos claros, tópicos bem organizados e tags relevantes.
5. Não inclua frontmatter YAML dentro de `content_markdown` (o sistema gera o frontmatter YAML automaticamente).
6. Use `path` relativo à raiz do cofre terminando em `.md` (sem `..`).
"""


class NoteUpdate(BaseModel):
    """Nota criada ou reorganizada no cofre Obsidian."""

    path: str
    title: str
    tags: List[str] = Field(default_factory=list)
    content_markdown: str
    links_added: List[str] = Field(default_factory=list)


class VaultOrganizationPlan(BaseModel):
    """Plano completo de organização e linkagem do Obsidian."""

    summary: str = ""
    notes: List[NoteUpdate] = Field(default_factory=list)


class ObsidianOrganizer:
    """Delega a organização semântica e linkagem do cofre Obsidian para um modelo de texto Gemini."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        fallback_model: str = FALLBACK_ORGANIZER_MODEL,
    ) -> None:
        env_model = os.getenv("OBSIDIAN_ORGANIZER_MODEL", "").strip()
        self.model_name = model_name or env_model or DEFAULT_ORGANIZER_MODEL
        self.fallback_model = fallback_model

    @staticmethod
    def _collect_vault_notes(
        vault_path: Path, max_files: int = 40, max_chars_per_file: int = 2500
    ) -> List[Dict[str, str]]:
        notes: List[Dict[str, str]] = []
        for file_path in sorted(vault_path.rglob("*.md")):
            if not file_path.is_file():
                continue
            rel = file_path.relative_to(vault_path)
            if any(part.startswith(".") for part in rel.parts):
                continue
            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception:  # noqa: BLE001
                continue
            notes.append(
                {
                    "path": rel.as_posix(),
                    "content": text[:max_chars_per_file],
                }
            )
            if len(notes) >= max_files:
                break
        return notes

    @staticmethod
    def _safe_target_path(vault_path: Path, rel_path: str) -> Optional[Path]:
        cleaned = rel_path.strip().replace("\\", "/").lstrip("/")
        if not cleaned or not cleaned.lower().endswith(".md"):
            return None
        vault_resolved = vault_path.resolve()
        candidate = (vault_resolved / cleaned).resolve()
        try:
            if not candidate.is_relative_to(vault_resolved):
                return None
        except AttributeError:
            if str(vault_resolved) not in str(candidate):
                return None
        if any(part.startswith(".") for part in candidate.relative_to(vault_resolved).parts):
            return None
        return candidate

    @staticmethod
    def _format_note_markdown(note: NoteUpdate) -> str:
        now_iso = datetime.now().isoformat(timespec="seconds")
        clean_title = (note.title or Path(note.path).stem).strip()
        tags = [t.strip().lstrip("#") for t in note.tags if t.strip()]
        if "hermes" not in tags:
            tags.append("hermes")
        tags_yaml = "\n".join(f"  - {t}" for t in tags)
        body = note.content_markdown.strip()
        # Remove possível frontmatter duplicado se o modelo tiver incluído mesmo assim.
        if body.startswith("---\n"):
            end_idx = body.find("\n---\n", 4)
            if end_idx != -1:
                body = body[end_idx + 5 :].lstrip()
        safe_title = clean_title.replace('"', "'")
        return (
            f"---\n"
            f'title: "{safe_title}"\n'
            f"updated_at: {now_iso}\n"
            f"tags:\n"
            f"{tags_yaml}\n"
            f"---\n\n"
            f"{body}\n"
        )

    def _generate_plan(
        self,
        genai_client: Any,
        notes: List[Dict[str, str]],
        focus: str,
    ) -> VaultOrganizationPlan:
        from google.genai import types

        notes_dump = "\n\n".join(
            f"=== ARQUIVO: {n['path']} ===\n{n['content']}" for n in notes
        )
        focus_line = f"\nPedido específico do usuário: {focus.strip()}\n" if focus.strip() else ""
        prompt = f"{_ORGANIZER_PROMPT}{focus_line}\nNotas atuais do cofre:\n{notes_dump}"
        cfg = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=VaultOrganizationPlan,
        )

        models_to_try = [self.model_name]
        if self.fallback_model and self.fallback_model not in models_to_try:
            models_to_try.append(self.fallback_model)

        last_exc: Optional[Exception] = None
        for model in models_to_try:
            try:
                resp = genai_client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=cfg,
                )
                parsed = getattr(resp, "parsed", None)
                if isinstance(parsed, VaultOrganizationPlan):
                    return parsed
                raw_text = getattr(resp, "text", "") or ""
                if raw_text.strip():
                    return VaultOrganizationPlan.model_validate_json(raw_text)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning("Organizador Obsidian falhou no modelo %s: %s", model, exc)
        if last_exc is not None:
            raise last_exc
        return VaultOrganizationPlan(summary="Nenhuma alteração gerada.")

    def organize(
        self,
        vault_path: Optional[Path],
        genai_client: Optional[Any],
        sqlite_store: Optional[EpisodicMemoryStore] = None,
        focus: str = "",
    ) -> str:
        """Lê o cofre, gera o plano no modelo de texto, aplica mudanças com segurança e reindexa FTS5."""
        if vault_path is None or not vault_path.is_dir():
            return "Cofre Obsidian não configurado ou pasta não encontrada."
        if genai_client is None or not hasattr(genai_client, "models"):
            return "Cliente da API Gemini indisponível para organizar o Obsidian."

        notes = self._collect_vault_notes(vault_path)
        if not notes:
            return "Não encontrei notas Markdown no cofre Obsidian para organizar."

        try:
            plan = self._generate_plan(genai_client, notes, focus)
        except Exception as exc:  # noqa: BLE001
            return f"Não consegui organizar o Obsidian agora por erro no modelo de texto: {exc}"

        updated_paths: List[str] = []
        for note in plan.notes:
            target = self._safe_target_path(vault_path, note.path)
            if target is None or not note.content_markdown.strip():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            full_md = self._format_note_markdown(note)
            target.write_text(full_md, encoding="utf-8")
            rel_posix = target.relative_to(vault_path.resolve()).as_posix()
            updated_paths.append(rel_posix)
            if sqlite_store is not None:
                sqlite_store.upsert_knowledge_doc(
                    path=rel_posix,
                    title=note.title or target.stem,
                    content=full_md,
                )

        if not updated_paths:
            return plan.summary or "O cofre já estava organizado e nenhuma nota precisou ser alterada."

        summary = plan.summary.strip() or f"Organizei e linkei {len(updated_paths)} nota(s) no seu Obsidian."
        return f"{summary} Arquivos atualizados: {', '.join(updated_paths[:8])}."
