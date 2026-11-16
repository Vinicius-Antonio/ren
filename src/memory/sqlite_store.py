"""Persistência SQLite (`memorias_episodicas`, `sessoes_historico`, `transcricoes`, `knowledge_fts`) e Consolidação Pós-Sessão."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH: Path = PROJECT_ROOT / "data" / "ren_memory.db"
VALID_CATEGORIES: Tuple[str, ...] = ("preferencia", "projeto", "rotina")

_EXTRACTION_PROMPT: str = """\
Você é o extrator de memória episódica do assistente Hermes.
Analise a transcrição abaixo e compare com as memórias já salvas (com seus IDs).
Gere um plano estruturado de operações:
- ADD: para fatos duradouros novos (preferências, localização, projetos, rotina).
- UPDATE: para atualizar uma memória existente informando o `id` dela e o novo `conteudo`.
- DELETE: para remover uma memória existente informando o `id` dela quando o fato não for mais válido.
Evite duplicatas de fatos que já existem. Inclua também um resumo curto da sessão (`session_summary`) e tópicos principais (`topics`).
"""


class MemoryOperation(BaseModel):
    """Operação estruturada de manutenção de memória episódica."""

    op: Literal["ADD", "UPDATE", "DELETE"]
    id: Optional[int] = None
    categoria: Literal["preferencia", "projeto", "rotina"] = "preferencia"
    conteudo: str = ""


class ConsolidationPlan(BaseModel):
    """Plano estruturado retornado pelo Gemini para consolidação pós-sessão."""

    operations: List[MemoryOperation] = Field(default_factory=list)
    session_summary: str = ""
    topics: List[str] = Field(default_factory=list)


class EpisodicMemoryStore:
    """Banco SQLite com memórias, sessões, transcrições e índice FTS5 sincronizado por triggers."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        self._last_consolidation_plan: Optional[ConsolidationPlan] = None
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=5.0)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _ensure_column(
        conn: sqlite3.Connection, table: str, column: str, definition: str
    ) -> None:
        cols = {
            row["name"]
            for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memorias_episodicas (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    categoria TEXT NOT NULL,
                    conteudo TEXT NOT NULL,
                    criado_em TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessoes_historico (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    resumo TEXT NOT NULL,
                    rolling_summary TEXT NOT NULL DEFAULT '',
                    turnos_json TEXT NOT NULL,
                    criado_em TEXT NOT NULL
                )
                """
            )
            self._ensure_column(
                conn,
                "sessoes_historico",
                "rolling_summary",
                "TEXT NOT NULL DEFAULT ''",
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS transcricoes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER,
                    speaker TEXT NOT NULL DEFAULT '',
                    user_text TEXT NOT NULL DEFAULT '',
                    assistant_text TEXT NOT NULL DEFAULT '',
                    compacted INTEGER NOT NULL DEFAULT 0,
                    criado_em TEXT NOT NULL
                )
                """
            )
            self._ensure_column(
                conn,
                "transcricoes",
                "compacted",
                "INTEGER NOT NULL DEFAULT 0",
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_docs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT NOT NULL UNIQUE,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
                    title,
                    content,
                    path UNINDEXED,
                    content='knowledge_docs',
                    content_rowid='id'
                )
                """
            )
            conn.executescript(
                """
                CREATE TRIGGER IF NOT EXISTS knowledge_docs_ai AFTER INSERT ON knowledge_docs BEGIN
                    INSERT INTO knowledge_fts(rowid, title, content, path)
                    VALUES (new.id, new.title, new.content, new.path);
                END;

                CREATE TRIGGER IF NOT EXISTS knowledge_docs_ad AFTER DELETE ON knowledge_docs BEGIN
                    INSERT INTO knowledge_fts(knowledge_fts, rowid, title, content, path)
                    VALUES ('delete', old.id, old.title, old.content, old.path);
                END;

                CREATE TRIGGER IF NOT EXISTS knowledge_docs_au AFTER UPDATE ON knowledge_docs BEGIN
                    INSERT INTO knowledge_fts(knowledge_fts, rowid, title, content, path)
                    VALUES ('delete', old.id, old.title, old.content, old.path);
                    INSERT INTO knowledge_fts(rowid, title, content, path)
                    VALUES (new.id, new.title, new.content, new.path);
                END;
                """
            )
            conn.commit()

    # ------------------------------------------------------- Memórias Episódicas
    def add_episodic_memory(
        self,
        conteudo: str,
        categoria: str = "preferencia",
        criado_em: Optional[str] = None,
    ) -> Optional[int]:
        """Insere um fato em `memorias_episodicas` evitando duplicatas case-insensitive."""
        cleaned = " ".join(conteudo.split()).strip()
        if not cleaned:
            return None
        cat = categoria.strip().lower()
        if cat not in VALID_CATEGORIES:
            cat = "preferencia"
        ts = criado_em or datetime.now().isoformat(timespec="seconds")

        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM memorias_episodicas WHERE LOWER(conteudo) = LOWER(?)",
                (cleaned,),
            ).fetchone()
            if row is not None:
                return int(row["id"])
            cur = conn.execute(
                """
                INSERT INTO memorias_episodicas (categoria, conteudo, criado_em)
                VALUES (?, ?, ?)
                """,
                (cat, cleaned, ts),
            )
            conn.commit()
            return int(cur.lastrowid or 0)

    def update_episodic_memory(
        self,
        memory_id: int,
        conteudo: str,
        categoria: Optional[str] = None,
    ) -> bool:
        """Atualiza uma memória existente pelo ID, evitando duplicatas."""
        cleaned = " ".join(conteudo.split()).strip()
        if not cleaned or memory_id <= 0:
            return False
        with self._lock, self._connect() as conn:
            dup = conn.execute(
                "SELECT id FROM memorias_episodicas WHERE LOWER(conteudo) = LOWER(?) AND id != ?",
                (cleaned, memory_id),
            ).fetchone()
            if dup is not None:
                return False
            if categoria is not None:
                cat = categoria.strip().lower()
                if cat not in VALID_CATEGORIES:
                    cat = "preferencia"
                cur = conn.execute(
                    "UPDATE memorias_episodicas SET conteudo = ?, categoria = ? WHERE id = ?",
                    (cleaned, cat, memory_id),
                )
            else:
                cur = conn.execute(
                    "UPDATE memorias_episodicas SET conteudo = ? WHERE id = ?",
                    (cleaned, memory_id),
                )
            conn.commit()
            return bool(cur.rowcount)

    def delete_episodic_memory_by_id(self, memory_id: int) -> bool:
        """Remove uma memória episódica pelo seu ID."""
        if memory_id <= 0:
            return False
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM memorias_episodicas WHERE id = ?",
                (memory_id,),
            )
            conn.commit()
            return bool(cur.rowcount)

    def delete_episodic_memory(self, query: str) -> int:
        term = query.strip().lower()
        if not term:
            return 0
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM memorias_episodicas WHERE LOWER(conteudo) LIKE ?",
                (f"%{term}%",),
            )
            conn.commit()
            return int(cur.rowcount or 0)

    def list_episodic_memories(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, categoria, conteudo, criado_em
                FROM memorias_episodicas
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def list_facts_formatted(self, limit: int = 20) -> List[str]:
        return [
            f"[{m['categoria']}] {m['conteudo']}"
            for m in self.list_episodic_memories(limit=limit)
        ]

    # ---------------------------------------------- Sessões & Transcrições
    def record_session_history(
        self,
        turns: Sequence[Dict[str, str]],
        resumo: str = "",
        rolling_summary: str = "",
        criado_em: Optional[str] = None,
    ) -> int:
        ts = criado_em or datetime.now().isoformat(timespec="seconds")
        if not resumo and turns:
            snippets = [
                f"U: {t.get('user', '')[:60]} -> H: {t.get('assistant', '')[:80]}"
                for t in turns[-4:]
                if t.get("user") or t.get("assistant")
            ]
            resumo = " | ".join(snippets)
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO sessoes_historico (resumo, rolling_summary, turnos_json, criado_em)
                VALUES (?, ?, ?, ?)
                """,
                (
                    resumo,
                    rolling_summary,
                    json.dumps(list(turns), ensure_ascii=False),
                    ts,
                ),
            )
            conn.commit()
            return int(cur.lastrowid or 0)

    def update_session_rolling_summary(
        self,
        session_id: Optional[int],
        rolling_summary: str,
    ) -> int:
        """Atualiza `rolling_summary` da sessão informada ou cria um novo registro de sessão."""
        ts = datetime.now().isoformat(timespec="seconds")
        with self._lock, self._connect() as conn:
            if session_id is not None and session_id > 0:
                cur = conn.execute(
                    "UPDATE sessoes_historico SET rolling_summary = ? WHERE id = ?",
                    (rolling_summary, session_id),
                )
                if cur.rowcount:
                    conn.commit()
                    return session_id
            cur = conn.execute(
                """
                INSERT INTO sessoes_historico (resumo, rolling_summary, turnos_json, criado_em)
                VALUES (?, ?, '[]', ?)
                """,
                ("", rolling_summary, ts),
            )
            conn.commit()
            return int(cur.lastrowid or 0)

    def get_latest_rolling_summary(self, session_id: Optional[int] = None) -> str:
        with self._lock, self._connect() as conn:
            if session_id is not None and session_id > 0:
                row = conn.execute(
                    "SELECT rolling_summary FROM sessoes_historico WHERE id = ?",
                    (session_id,),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT rolling_summary FROM sessoes_historico
                    WHERE rolling_summary != ''
                    ORDER BY id DESC LIMIT 1
                    """
                ).fetchone()
            return str(row["rolling_summary"]) if row is not None else ""

    def list_session_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, resumo, rolling_summary, turnos_json, criado_em
                FROM sessoes_historico
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def add_transcript_turn(
        self,
        user_text: str,
        assistant_text: str,
        speaker: str = "",
        session_id: Optional[int] = None,
        criado_em: Optional[str] = None,
    ) -> int:
        ts = criado_em or datetime.now().isoformat(timespec="seconds")
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO transcricoes (session_id, speaker, user_text, assistant_text, compacted, criado_em)
                VALUES (?, ?, ?, ?, 0, ?)
                """,
                (session_id, speaker, user_text, assistant_text, ts),
            )
            conn.commit()
            return int(cur.lastrowid or 0)

    def get_uncompacted_turns(
        self, session_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            if session_id is not None:
                rows = conn.execute(
                    """
                    SELECT id, session_id, speaker, user_text, assistant_text, compacted, criado_em
                    FROM transcricoes
                    WHERE compacted = 0 AND session_id = ?
                    ORDER BY id ASC
                    """,
                    (session_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, session_id, speaker, user_text, assistant_text, compacted, criado_em
                    FROM transcricoes
                    WHERE compacted = 0
                    ORDER BY id ASC
                    """
                ).fetchall()
        return [dict(r) for r in rows]

    def mark_turns_compacted(self, turn_ids: Sequence[int]) -> int:
        ids = [int(i) for i in turn_ids if int(i) > 0]
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                f"UPDATE transcricoes SET compacted = 1 WHERE id IN ({placeholders})",
                ids,
            )
            conn.commit()
            return int(cur.rowcount or 0)

    # ------------------------------------------------------- Índice FTS5
    def upsert_knowledge_doc(
        self,
        path: str,
        title: str,
        content: str,
        updated_at: Optional[str] = None,
    ) -> int:
        """Insere ou atualiza um documento em `knowledge_docs` (sincronizado via triggers com `knowledge_fts`)."""
        ts = updated_at or datetime.now().isoformat(timespec="seconds")
        with self._lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT id FROM knowledge_docs WHERE path = ?",
                (path,),
            ).fetchone()
            if existing is not None:
                doc_id = int(existing["id"])
                conn.execute(
                    """
                    UPDATE knowledge_docs
                    SET title = ?, content = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (title, content, ts, doc_id),
                )
            else:
                cur = conn.execute(
                    """
                    INSERT INTO knowledge_docs (path, title, content, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (path, title, content, ts),
                )
                doc_id = int(cur.lastrowid or 0)
            conn.commit()
            return doc_id

    @staticmethod
    def _build_fts_query(query: str) -> str:
        tokens = re.findall(r"[0-9A-Za-zÀ-ÿ_]+", query)
        if not tokens:
            return ""
        return " OR ".join(f'"{t}"' for t in tokens)

    def search_knowledge_fts(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Busca documentos indexados no `knowledge_fts` ordenados por relevância BM25."""
        fts_query = self._build_fts_query(query)
        if not fts_query:
            return []
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    d.id,
                    d.path,
                    d.title,
                    snippet(knowledge_fts, 1, '[', ']', '...', 24) AS snippet,
                    d.content,
                    bm25(knowledge_fts) AS score
                FROM knowledge_fts
                JOIN knowledge_docs d ON d.id = knowledge_fts.rowid
                WHERE knowledge_fts MATCH ?
                ORDER BY score
                LIMIT ?
                """,
                (fts_query, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------ Pipeline de Consolidação Pós-Sessão
    def _apply_consolidation_plan(self, plan: ConsolidationPlan) -> List[Dict[str, str]]:
        applied: List[Dict[str, str]] = []
        for op in plan.operations:
            action = op.op.upper()
            if action == "ADD" and op.conteudo.strip():
                self.add_episodic_memory(op.conteudo, op.categoria)
                applied.append({"categoria": op.categoria, "conteudo": op.conteudo.strip()})
            elif action == "UPDATE" and op.id is not None and op.conteudo.strip():
                if self.update_episodic_memory(op.id, op.conteudo, op.categoria):
                    applied.append({"categoria": op.categoria, "conteudo": op.conteudo.strip()})
            elif action == "DELETE" and op.id is not None:
                self.delete_episodic_memory_by_id(op.id)
        return applied

    def consolidate_session(
        self,
        turns: Sequence[Dict[str, str]],
        genai_client: Optional[Any] = None,
        model_name: str = "gemini-2.5-flash",
    ) -> List[Dict[str, str]]:
        """Executa o pipeline pós-sessão: salva `sessoes_historico` e aplica ADD/UPDATE/DELETE via Pydantic."""
        valid_turns = [
            t
            for t in turns
            if (t.get("user", "").strip() or t.get("assistant", "").strip())
        ]
        if not valid_turns:
            return []

        self._last_consolidation_plan = None
        extracted: List[Dict[str, str]] = []
        rolling_summary = self.get_latest_rolling_summary()
        session_summary = ""

        # 1. Tenta extração estruturada via Gemini (Structured Outputs com Pydantic)
        if genai_client is not None and hasattr(genai_client, "models"):
            try:
                from google.genai import types

                existing = self.list_episodic_memories(limit=50)
                existing_lines = (
                    "\n".join(
                        f"- id={m['id']} [{m['categoria']}] {m['conteudo']}"
                        for m in existing
                    )
                    or "(nenhuma memória salva ainda)"
                )
                transcript_lines = [
                    f"Usuário: {t.get('user', '')}\nHermes: {t.get('assistant', '')}"
                    for t in valid_turns
                ]
                prompt = (
                    f"{_EXTRACTION_PROMPT}\n"
                    f"Memórias existentes:\n{existing_lines}\n\n"
                    f"Transcrição:\n" + "\n".join(transcript_lines)
                )
                cfg = types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ConsolidationPlan,
                )
                resp = genai_client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=cfg,
                )
                parsed = getattr(resp, "parsed", None)
                plan: Optional[ConsolidationPlan] = None
                if isinstance(parsed, ConsolidationPlan):
                    plan = parsed
                else:
                    raw_text = getattr(resp, "text", "") or ""
                    if raw_text.strip().startswith("{"):
                        plan = ConsolidationPlan.model_validate_json(raw_text)
                    else:
                        match = re.search(r"\[.*\]", raw_text, flags=re.DOTALL)
                        if match:
                            items = json.loads(match.group(0))
                            ops = [
                                MemoryOperation(
                                    op="ADD",
                                    categoria=str(i.get("categoria", "preferencia")).lower()
                                    if str(i.get("categoria", "preferencia")).lower() in VALID_CATEGORIES
                                    else "preferencia",
                                    conteudo=str(i["conteudo"]).strip(),
                                )
                                for i in items
                                if isinstance(i, dict) and i.get("conteudo")
                            ]
                            plan = ConsolidationPlan(operations=ops)

                if plan is not None:
                    self._last_consolidation_plan = plan
                    session_summary = plan.session_summary.strip()
                    extracted = self._apply_consolidation_plan(plan)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Fallback local na consolidação pós-sessão: %s", exc)

        # 2. Extrator heurístico determinístico de garantia (offline / sem custo extra)
        if not extracted and self._last_consolidation_plan is None:
            for item in self._heuristic_extract(valid_turns):
                self.add_episodic_memory(item["conteudo"], item["categoria"])
                extracted.append(item)

        self.record_session_history(
            valid_turns,
            resumo=session_summary,
            rolling_summary=rolling_summary,
        )
        return extracted

    @staticmethod
    def _heuristic_extract(turns: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
        """Extrai declarações explícitas de preferência, projeto ou rotina da fala do usuário."""
        found: List[Dict[str, str]] = []
        patterns: Tuple[Tuple[re.Pattern[str], str], ...] = (
            (
                re.compile(
                    r"\b(eu moro em [^.?!]+|meu nome e [^.?!]+|eu gosto de [^.?!]+|eu prefiro [^.?!]+)",
                    re.IGNORECASE,
                ),
                "preferencia",
            ),
            (
                re.compile(
                    r"\b(estou trabalhando no projeto [^.?!]+|meu projeto [^.?!]+|estou desenvolvendo [^.?!]+)",
                    re.IGNORECASE,
                ),
                "projeto",
            ),
            (
                re.compile(
                    r"\b(todo dia eu [^.?!]+|minha rotina [^.?!]+|sempre acordo [^.?!]+)",
                    re.IGNORECASE,
                ),
                "rotina",
            ),
        )
        for t in turns:
            u = t.get("user", "").strip()
            if not u:
                continue
            for regex, cat in patterns:
                m = regex.search(u)
                if m:
                    found.append({"categoria": cat, "conteudo": m.group(1).strip()})
        return found
