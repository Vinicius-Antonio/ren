"""Gerenciador de Memória de Longo Prazo, Histórico Compacto e Base de Conhecimento Local."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import PROJECT_ROOT
from src.core.router import SlowLoopManager
from src.discord_bot.bot import DiscordVoiceBot
from src.gateway.ws_server import GatewayServer
from src.memory.obsidian_organizer import ObsidianOrganizer
from src.memory.sqlite_store import EpisodicMemoryStore
from src.system.briefing import build_executive_briefing
from src.system.email_client import read_recent_emails, search_emails
from src.system.pc_actions import (
    control_media,
    get_pc_status,
    open_application,
    open_website_or_search,
    play_music,
    read_clipboard,
    write_clipboard,
)
from src.system.timers import TimerManager

logger = logging.getLogger(__name__)

DEFAULT_MEMORY_FILE: Path = PROJECT_ROOT / "data" / "memory.json"
DEFAULT_KNOWLEDGE_DIR: Path = PROJECT_ROOT / "knowledge"

# Limites estritos para garantir que o prompt nunca fique grande (< 350 tokens)
MAX_FACTS: int = 10
MAX_FACT_CHARS: int = 160
MAX_RECENT_TURNS: int = 4
MAX_TURN_CHARS: int = 180
MAX_SUMMARY_CHARS: int = 500
RECENT_TURN_EXPIRY_HOURS: float = 6.0


def _compact_text(text: str, max_chars: int) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 1].rstrip() + "…"


class KnowledgeManager:
    """Gerencia memórias (`data/memory.json`), histórico reduzido de conversas e `knowledge/`."""

    def __init__(
        self,
        memory_file: Path = DEFAULT_MEMORY_FILE,
        knowledge_dir: Path = DEFAULT_KNOWLEDGE_DIR,
        timer_manager: Optional[TimerManager] = None,
        sqlite_store: Optional[EpisodicMemoryStore] = None,
        gateway: Optional[GatewayServer] = None,
        slow_loop: Optional[SlowLoopManager] = None,
        discord_bot: Optional[DiscordVoiceBot] = None,
        obsidian_vault_path: Optional[Path] = None,
        genai_client: Optional[Any] = None,
    ) -> None:
        self.memory_file = memory_file
        self.knowledge_dir = knowledge_dir
        self.timers = timer_manager or TimerManager()
        db_path = (
            PROJECT_ROOT / "data" / "ren_memory.db"
            if memory_file == DEFAULT_MEMORY_FILE
            else memory_file.with_suffix(".db")
        )
        self.sqlite_store = sqlite_store or EpisodicMemoryStore(db_path=db_path)
        self.gateway = gateway
        self.slow_loop = slow_loop or SlowLoopManager(gateway=gateway)
        self.discord_bot = discord_bot
        self.genai_client = genai_client
        self._organizer = ObsidianOrganizer()
        env_vault = os.getenv("OBSIDIAN_VAULT_PATH", "").strip()
        self.obsidian_vault_path: Optional[Path] = (
            obsidian_vault_path
            if obsidian_vault_path is not None
            else (Path(env_vault) if env_vault else None)
        )
        self._session_turns: List[Dict[str, str]] = []
        self._last_vault_index_ts: float = 0.0

    def _load_store(self) -> Dict[str, Any]:
        default_store: Dict[str, Any] = {
            "memories": [],
            "rolling_summary": "",
            "recent_turns": [],
        }
        if not self.memory_file.is_file():
            return default_store
        try:
            raw = json.loads(self.memory_file.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return default_store
            return {
                "memories": list(raw.get("memories", [])),
                "rolling_summary": str(raw.get("rolling_summary", "")),
                "recent_turns": list(raw.get("recent_turns", [])),
            }
        except Exception:  # noqa: BLE001
            logger.warning("Falha ao ler %s", self.memory_file, exc_info=True)
            return default_store

    def _save_store(self, store: Dict[str, Any]) -> None:
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        # Substituição atômica: uma interrupção não deixa JSON pela metade.
        temp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.memory_file.parent,
                prefix=".memory-", suffix=".tmp", delete=False,
            ) as temp_file:
                temp_path = Path(temp_file.name)
                json.dump(store, temp_file, ensure_ascii=False, indent=2)
                temp_file.flush()
                os.fsync(temp_file.fileno())
            os.replace(temp_path, self.memory_file)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    # ----------------------------------------------------------------- memória
    def list_memories(self) -> List[str]:
        store = self._load_store()
        return [
            str(item["fact"])
            for item in store["memories"][-MAX_FACTS:]
            if isinstance(item, dict) and "fact" in item
        ]

    def save_memory(self, fact: str, categoria: str = "preferencia") -> str:
        cleaned = _compact_text(fact, MAX_FACT_CHARS)
        if not cleaned:
            return "Nenhum fato informado."
        store = self._load_store()
        existing: List[Dict[str, str]] = store["memories"]

        # Evita duplicatas exatas (case-insensitive)
        if not any(m.get("fact", "").lower() == cleaned.lower() for m in existing):
            existing.append(
                {
                    "fact": cleaned,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
            store["memories"] = existing[-MAX_FACTS:]
            self._save_store(store)
        self.sqlite_store.add_episodic_memory(cleaned, categoria=categoria)
        return f"Memória salva com sucesso: {cleaned}"

    def delete_memory(self, query: str) -> str:
        term = query.strip().lower()
        if not term:
            return "Nenhuma memória encontrada para remover."
        self.sqlite_store.delete_episodic_memory(term)
        if not self.memory_file.is_file():
            return "Nenhuma memória encontrada para remover."
        store = self._load_store()
        existing: List[Dict[str, str]] = store["memories"]

        kept = [m for m in existing if term not in m.get("fact", "").lower()]
        removed_count = len(existing) - len(kept)
        store["memories"] = kept
        self._save_store(store)
        return f"Removidas {removed_count} memória(s) contendo '{query}'."

    def _write_obsidian_voice_log(
        self,
        turns: List[Dict[str, str]],
        extracted: List[Dict[str, str]],
    ) -> Optional[Path]:
        """Gera ou atualiza `VoiceLogs/YYYY-MM-DD - Resumo.md` no cofre Obsidian com [[wikilinks]]."""
        if self.obsidian_vault_path is None:
            return None
        logs_dir = self.obsidian_vault_path / "VoiceLogs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M")
        log_path = logs_dir / f"{date_str} - Resumo.md"

        plan = getattr(self.sqlite_store, "_last_consolidation_plan", None)
        summary = (
            plan.session_summary.strip()
            if plan is not None and getattr(plan, "session_summary", "").strip()
            else f"Sessão de voz com {len(turns)} turno(s) registrada às {time_str}."
        )
        topics: List[str] = []
        if plan is not None and getattr(plan, "topics", None):
            topics = [str(t).strip() for t in plan.topics if str(t).strip()]
        if not topics:
            topics = ["Hermes", "VoiceNotes"]

        wikilinks = " ".join(f"[[{t}]]" for t in topics)
        facts_block = ""
        if extracted:
            facts_lines = "\n".join(
                f"- [[{item.get('categoria', 'preferencia')}]] {item.get('conteudo', '')}"
                for item in extracted
                if item.get("conteudo")
            )
            facts_block = f"\n### Memórias Consolidadas\n{facts_lines}\n"

        section = (
            f"\n## Sessão {time_str}\n"
            f"**Resumo:** {summary}\n\n"
            f"**Links:** {wikilinks}\n"
            f"{facts_block}"
        )
        if log_path.is_file():
            existing = log_path.read_text(encoding="utf-8", errors="ignore")
            log_path.write_text(existing.rstrip() + "\n" + section, encoding="utf-8")
        else:
            frontmatter = (
                f"---\n"
                f"date: {date_str}\n"
                f"tags:\n"
                f"  - voice-log\n"
                f"  - hermes\n"
                f"---\n\n"
                f"# Resumo de Voz — {date_str}\n"
            )
            log_path.write_text(frontmatter + section, encoding="utf-8")
        return log_path

    def consolidate_post_session(
        self, genai_client: Optional[Any] = None
    ) -> List[Dict[str, str]]:
        """Executa o pipeline pós-sessão extraindo fatos duradouros para o SQLite, JSON e Obsidian."""
        turns = list(self._session_turns)
        self._session_turns.clear()
        if not turns:
            return []
        extracted = self.sqlite_store.consolidate_session(
            turns=turns, genai_client=genai_client
        )
        for item in extracted:
            self.save_memory(item["conteudo"], categoria=item.get("categoria", "preferencia"))
        self._write_obsidian_voice_log(turns, extracted)
        return extracted

    # --------------------------------------- memória reduzida de conversas
    @staticmethod
    def _fold_turn_into_summary(
        current_summary: str, turn: Dict[str, str]
    ) -> str:
        """Comprime um turno antigo que saiu da janela em uma linha curta no resumo."""
        u = _compact_text(turn.get("user", ""), 70)
        a = _compact_text(turn.get("assistant", ""), 90)
        if u and a:
            bullet = f"Usuário perguntou/disse '{u}' -> Ren: '{a}'."
        elif u:
            bullet = f"Usuário disse '{u}'."
        elif a:
            bullet = f"Ren disse '{a}'."
        else:
            return current_summary

        combined = f"{current_summary} | {bullet}".strip(" |") if current_summary else bullet
        if len(combined) <= MAX_SUMMARY_CHARS:
            return combined
        # Mantém apenas a parte mais recente do resumo dentro do teto de caracteres
        trimmed = combined[-MAX_SUMMARY_CHARS:]
        sep = trimmed.find("| ")
        return trimmed[sep + 2 :].strip() if sep != -1 else trimmed.strip()

    def record_turn(
        self,
        user_text: str,
        assistant_text: str,
        now: Optional[datetime] = None,
        speaker_id: Optional[int] = None,
        speaker_name: str = "",
    ) -> None:
        """Registra um turno da conversa mantendo janela máxima fixa de 4 turnos + resumo."""
        u_clean = _compact_text(user_text, MAX_TURN_CHARS)
        a_clean = _compact_text(assistant_text, MAX_TURN_CHARS)
        if not u_clean and not a_clean:
            return
        # Ignora transcrições espúrias em alfabetos não-latinos (ruído interpretado como cirílico/asiático)
        if re.search(r"[\u0400-\u04FF\u4E00-\u9FFF\u3040-\u30FF]", u_clean):
            return

        ts = (now or datetime.now()).isoformat(timespec="seconds")
        turn = {"user": u_clean, "assistant": a_clean, "ts": ts}
        if speaker_id is not None:
            turn["speaker_id"] = str(speaker_id)
            turn["speaker_name"] = speaker_name
        self._session_turns.append(turn)
        self.sqlite_store.add_transcript_turn(
            user_text=u_clean,
            assistant_text=a_clean,
            speaker=speaker_name or ("Participante" if speaker_id is not None else "Usuário"),
            criado_em=ts,
        )
        store = self._load_store()
        turns: List[Dict[str, str]] = store["recent_turns"]
        summary: str = store["rolling_summary"]

        turns.append(turn.copy())

        while len(turns) > MAX_RECENT_TURNS:
            oldest = turns.pop(0)
            summary = self._fold_turn_into_summary(summary, oldest)

        store["recent_turns"] = turns
        store["rolling_summary"] = summary
        self._save_store(store)

    def get_conversation_context(
        self,
        now: Optional[datetime] = None,
        expiry_hours: float = RECENT_TURN_EXPIRY_HOURS,
    ) -> str:
        """Retorna o histórico reduzido pronto para o prompt (sempre compacto)."""
        store = self._load_store()
        turns: List[Dict[str, str]] = store["recent_turns"]
        summary: str = store["rolling_summary"]
        if not turns and not summary:
            return ""

        current = now or datetime.now()
        cutoff = current - timedelta(hours=expiry_hours)
        fresh_turns: List[Dict[str, str]] = []
        changed = False

        for t in turns:
            ts_raw = t.get("ts", "")
            is_expired = False
            if ts_raw:
                try:
                    dt = datetime.fromisoformat(ts_raw)
                    if dt.tzinfo is not None and current.tzinfo is None:
                        dt = dt.replace(tzinfo=None)
                    is_expired = dt < cutoff
                except ValueError:
                    is_expired = False
            if is_expired:
                summary = self._fold_turn_into_summary(summary, t)
                changed = True
            else:
                fresh_turns.append(t)

        if changed:
            store["recent_turns"] = fresh_turns
            store["rolling_summary"] = summary
            self._save_store(store)

        parts: List[str] = []
        if summary:
            parts.append(f"Resumo anterior: {summary}")
        if fresh_turns:
            lines = []
            for t in fresh_turns:
                u = t.get("user", "")
                a = t.get("assistant", "")
                speaker = t.get("speaker_name", "")
                speaker_id = t.get("speaker_id", "")
                label = f"{speaker or 'Participante'} (Discord {speaker_id})" if speaker_id else "Usuário"
                if u and a:
                    lines.append(f"- {label}: {u} | Ren: {a}")
                elif u:
                    lines.append(f"- {label}: {u}")
                elif a:
                    lines.append(f"- Ren: {a}")
            if lines:
                parts.append("Últimas falas recentes:\n" + "\n".join(lines))
        return "\n".join(parts)

    # ------------------------------------------------- base de conhecimento
    def search_local_knowledge(self, query: str, max_chars: int = 2500) -> str:
        """Busca trechos relevantes nos arquivos .md e .txt da pasta `knowledge/`."""
        if not self.knowledge_dir.is_dir():
            return "Pasta knowledge/ não encontrada."

        terms = [t.lower() for t in query.split() if len(t) >= 2]
        results: List[tuple[int, str, str]] = []

        for file_path in sorted(self.knowledge_dir.rglob("*")):
            if not file_path.is_file() or file_path.suffix.lower() not in {".md", ".txt"}:
                continue
            if file_path.name.lower() == "readme.md":
                continue
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception:  # noqa: BLE001
                continue
            if not content:
                continue
            lower_content = content.lower()
            score = sum(lower_content.count(t) for t in terms) + sum(
                3 for t in terms if t in file_path.stem.lower()
            )
            if score > 0 or not terms:
                results.append((score, file_path.name, content[:1200]))

        if not results:
            files = [
                p.name
                for p in sorted(self.knowledge_dir.rglob("*"))
                if p.is_file()
                and p.suffix.lower() in {".md", ".txt"}
                and p.name.lower() != "readme.md"
            ]
            if not files:
                return "Nenhum documento pessoal encontrado na pasta knowledge/."
            return f"Nenhum trecho exato para '{query}'. Arquivos disponíveis: {', '.join(files)}"

        results.sort(key=lambda item: item[0], reverse=True)
        formatted: List[str] = []
        total = 0
        for _, name, snippet in results:
            entry = f"[{name}]: {snippet}"
            if total + len(entry) > max_chars:
                break
            formatted.append(entry)
            total += len(entry)
        return "\n\n".join(formatted)

    def list_knowledge_files(self) -> List[str]:
        if not self.knowledge_dir.is_dir():
            return []
        return [
            p.name
            for p in sorted(self.knowledge_dir.rglob("*"))
            if p.is_file()
            and p.suffix.lower() in {".md", ".txt"}
            and p.name.lower() != "readme.md"
        ]

    def index_obsidian_vault(self) -> int:
        """Indexa todos os arquivos `.md` de `OBSIDIAN_VAULT_PATH` (UTF-8) no `knowledge_fts`."""
        if self.obsidian_vault_path is None or not self.obsidian_vault_path.is_dir():
            return 0
        count = 0
        for file_path in sorted(self.obsidian_vault_path.rglob("*.md")):
            if not file_path.is_file():
                continue
            rel = file_path.relative_to(self.obsidian_vault_path)
            if any(part.startswith(".") for part in rel.parts):
                continue
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception:  # noqa: BLE001
                continue
            if not content:
                continue
            title = file_path.stem
            for line in content.splitlines():
                stripped = line.strip()
                if stripped.startswith("# "):
                    title = stripped[2:].strip() or title
                    break
            self.sqlite_store.upsert_knowledge_doc(
                path=rel.as_posix(),
                title=title,
                content=content,
            )
            count += 1
        return count

    def search_obsidian_notes(self, query: str, limit: int = 5) -> str:
        """Indexa o cofre Obsidian (com cache de 60s) e pesquisa no índice FTS5."""
        import time

        if self.obsidian_vault_path is None or not self.obsidian_vault_path.is_dir():
            return "Cofre Obsidian não configurado ou não encontrado."
        now = time.monotonic()
        if now - self._last_vault_index_ts > 60.0:
            self.index_obsidian_vault()
            self._last_vault_index_ts = now
        hits = self.sqlite_store.search_knowledge_fts(query, limit=limit)
        if not hits:
            return f"Nenhuma nota encontrada no Obsidian para '{query}'."
        formatted = [
            f"[{hit['title']} ({hit['path']})]: {hit['snippet'] or str(hit['content'])[:300]}"
            for hit in hits
        ]
        return "\n\n".join(formatted)

    def organize_obsidian_vault(self, focus: str = "") -> str:
        """Delega a organização e linkagem (`[[wikilinks]]` / MOCs) do Obsidian ao modelo de texto Gemini."""
        import time

        res = self._organizer.organize(
            vault_path=self.obsidian_vault_path,
            genai_client=self.genai_client,
            sqlite_store=self.sqlite_store,
            focus=focus,
        )
        self._last_vault_index_ts = time.monotonic()
        return res

    def create_voice_note(self, title: str, content: str) -> str:
        """Cria ou adiciona conteúdo a uma nota Markdown (em `OBSIDIAN_VAULT_PATH/VoiceNotes/` ou `knowledge/`)."""
        clean_title = title.strip() or "Nota Rápida"
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", clean_title.lower()).strip("_") or "nota_rapida"
        now = datetime.now()

        if self.obsidian_vault_path is not None:
            notes_dir = self.obsidian_vault_path / "VoiceNotes"
            notes_dir.mkdir(parents=True, exist_ok=True)
            note_path = notes_dir / f"{slug}.md"
            iso_stamp = now.isoformat(timespec="seconds")
            human_stamp = now.strftime("%d/%m/%Y %H:%M")
            if note_path.is_file():
                existing = note_path.read_text(encoding="utf-8", errors="ignore")
                full_text = existing.rstrip() + f"\n\n## {clean_title} ({human_stamp})\n{content.strip()}\n"
            else:
                frontmatter = (
                    f"---\n"
                    f"title: \"{clean_title.replace(chr(34), chr(39))}\"\n"
                    f"date: {iso_stamp}\n"
                    f"tags:\n"
                    f"  - voice-note\n"
                    f"  - hermes\n"
                    f"---\n\n"
                )
                full_text = f"{frontmatter}# {clean_title}\n\n{content.strip()}\n"
            note_path.write_text(full_text, encoding="utf-8")
            self.sqlite_store.upsert_knowledge_doc(
                path=f"VoiceNotes/{note_path.name}",
                title=clean_title,
                content=full_text,
            )
            return f"Nota salva no Obsidian em VoiceNotes/{note_path.name}."

        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        note_path = self.knowledge_dir / f"{slug}.md"
        stamp = now.strftime("%d/%m/%Y %H:%M")
        entry = f"\n## {clean_title} ({stamp})\n{content.strip()}\n"
        if note_path.is_file():
            existing = note_path.read_text(encoding="utf-8", errors="ignore")
            note_path.write_text(existing.rstrip() + "\n" + entry, encoding="utf-8")
        else:
            note_path.write_text(f"# {title.strip() or 'Notas'}\n" + entry, encoding="utf-8")
        return f"Nota salva em knowledge/{note_path.name}."

    # ------------------------------------------------ integração Gemini Tools
    def build_function_declarations(self) -> List[Any]:
        from google.genai import types

        declarations = [
            types.FunctionDeclaration(
                name="save_memory",
                description=(
                    "Salva um fato importante sobre o usuário (nome, cidade, "
                    "preferências, projetos, rotina ou lembretes) na memória permanente."
                ),
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "fact": types.Schema(
                            type="STRING",
                            description="Fato claro e conciso em português para lembrar no futuro.",
                        )
                    },
                    required=["fact"],
                ),
            ),
            types.FunctionDeclaration(
                name="delete_memory",
                description="Remove um fato da memória permanente quando o usuário pedir para esquecer algo.",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "query": types.Schema(
                            type="STRING",
                            description="Palavra-chave da memória que deve ser apagada.",
                        )
                    },
                    required=["query"],
                ),
            ),
            types.FunctionDeclaration(
                name="search_local_knowledge",
                description=(
                    "Pesquisa nos documentos, notas e arquivos locais do usuário "
                    "dentro da pasta knowledge/."
                ),
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "query": types.Schema(
                            type="STRING",
                            description="Termos de busca para procurar nos arquivos locais.",
                        )
                    },
                    required=["query"],
                ),
            ),
            types.FunctionDeclaration(
                name="look_at_screen",
                description=(
                    "Captura e enxerga a tela atual do computador do usuário em tempo real. "
                    "Chame SEMPRE que o usuário pedir para olhar a tela, analisar um código, "
                    "explicar um erro, ler ou traduzir o que está visível no monitor."
                ),
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "focus": types.Schema(
                            type="STRING",
                            description="O que observar ou procurar na tela do usuário.",
                        )
                    },
                ),
            ),
            types.FunctionDeclaration(
                name="play_music",
                description=(
                    "Toca uma música, artista, álbum ou playlist no Spotify Desktop "
                    "ou no YouTube."
                ),
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "query": types.Schema(
                            type="STRING",
                            description="Nome da música, artista ou estilo musical para tocar.",
                        ),
                        "platform": types.Schema(
                            type="STRING",
                            description="Plataforma alvo: 'spotify' (padrão) ou 'youtube'.",
                        ),
                    },
                    required=["query"],
                ),
            ),
            types.FunctionDeclaration(
                name="control_media",
                description=(
                    "Controla reprodução de música/mídia (play_pause, next, previous, stop) "
                    "e o volume do Windows (volume_up, volume_down, mute)."
                ),
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "action": types.Schema(
                            type="STRING",
                            description=(
                                "Ação: 'play_pause', 'next', 'previous', 'stop', "
                                "'volume_up', 'volume_down' ou 'mute'."
                            ),
                        ),
                        "steps": types.Schema(
                            type="INTEGER",
                            description="Quantidade de passos de volume (padrão: 5).",
                        ),
                    },
                    required=["action"],
                ),
            ),
            types.FunctionDeclaration(
                name="open_application",
                description=(
                    "Abre um aplicativo instalado no Windows (Spotify, Calculadora, "
                    "Bloco de Notas, Explorador de Arquivos, Paint, Terminal, VS Code, Configurações)."
                ),
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "app_name": types.Schema(
                            type="STRING",
                            description="Nome do aplicativo para iniciar.",
                        )
                    },
                    required=["app_name"],
                ),
            ),
            types.FunctionDeclaration(
                name="open_website_or_search",
                description=(
                    "Abre um site no navegador (youtube, github, gmail, whatsapp, google, etc.) "
                    "ou faz uma pesquisa aberta no navegador."
                ),
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "target": types.Schema(
                            type="STRING",
                            description="Nome do site ou URL (ex: 'youtube', 'github', 'google').",
                        ),
                        "search_query": types.Schema(
                            type="STRING",
                            description="Termo opcional para pesquisar dentro do site.",
                        ),
                    },
                    required=["target"],
                ),
            ),
            types.FunctionDeclaration(
                name="read_clipboard",
                description=(
                    "Lê o texto copiado atualmente na área de transferência (Ctrl+C) do Windows."
                ),
                parameters=types.Schema(type="OBJECT", properties={}),
            ),
            types.FunctionDeclaration(
                name="write_clipboard",
                description=(
                    "Copia um texto ou código diretamente para a área de transferência (Ctrl+V) do usuário."
                ),
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "text": types.Schema(
                            type="STRING",
                            description="Texto ou código que será copiado para o Ctrl+V.",
                        )
                    },
                    required=["text"],
                ),
            ),
            types.FunctionDeclaration(
                name="get_pc_status",
                description=(
                    "Consulta o uso real de memória RAM e espaço livre em disco do computador."
                ),
                parameters=types.Schema(type="OBJECT", properties={}),
            ),
            types.FunctionDeclaration(
                name="get_executive_briefing",
                description=(
                    "Gera o 'Briefing Executivo' completo em tempo real (clima local e chuva, "
                    "cotação do dólar/euro/bitcoin, as 3 principais notícias do momento no Brasil, "
                    "status do PC, e-mails não lidos e lembretes salvos). Chame SEMPRE que o usuário disser "
                    "'Bom dia', 'Boa tarde', 'Boa noite', 'Me atualiza' ou pedir um briefing/resumo do dia."
                ),
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "city": types.Schema(
                            type="STRING",
                            description="Cidade do usuário, se conhecida na memória (opcional; se vazio, detecta automaticamente).",
                        )
                    },
                ),
            ),
            types.FunctionDeclaration(
                name="read_recent_emails",
                description=(
                    "Lê os e-mails mais recentes da caixa de entrada do usuário (não lidos ou todos)."
                ),
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "unread_only": types.Schema(
                            type="BOOLEAN",
                            description="True para ler apenas não lidos; False para ler os últimos recebidos.",
                        ),
                        "limit": types.Schema(
                            type="INTEGER",
                            description="Quantidade máxima de e-mails para listar (padrão: 4).",
                        ),
                    },
                ),
            ),
            types.FunctionDeclaration(
                name="search_emails",
                description=(
                    "Pesquisa na caixa de e-mail do usuário por remetente, assunto ou palavra-chave "
                    "(ex: código de verificação, boleto, Amazon, faculdade)."
                ),
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "query": types.Schema(
                            type="STRING",
                            description="Termo de busca (assunto ou remetente).",
                        )
                    },
                    required=["query"],
                ),
            ),
            types.FunctionDeclaration(
                name="set_timer",
                description=(
                    "Programa um timer, alarme ou lembrete falado para daqui a N segundos/minutos "
                    "(ex: 'daqui a 5 minutos me lembre que devo fazer tal coisa'). "
                    "Quando o tempo acabar, o Ren acorda sozinho e fala o lembrete em voz alta."
                ),
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "duration_seconds": types.Schema(
                            type="INTEGER",
                            description="Duração total em segundos (ex: 300 para 5 minutos, 600 para 10 minutos).",
                        ),
                        "label": types.Schema(
                            type="STRING",
                            description="O que o Ren deve falar/lembrar ao usuário quando o tempo acabar.",
                        ),
                    },
                    required=["duration_seconds", "label"],
                ),
            ),
            types.FunctionDeclaration(
                name="create_voice_note",
                description=(
                    "Cria ou salva uma anotação em arquivo Markdown diretamente no cofre Obsidian "
                    "(pasta VoiceNotes/ de OBSIDIAN_VAULT_PATH, com frontmatter YAML; ou knowledge/ se o cofre não estiver configurado)."
                ),
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "title": types.Schema(
                            type="STRING",
                            description="Título curto da nota.",
                        ),
                        "content": types.Schema(
                            type="STRING",
                            description="Conteúdo completo da anotação.",
                        ),
                    },
                    required=["title", "content"],
                ),
            ),
            types.FunctionDeclaration(
                name="join_discord_call",
                description=(
                    "Conecta o Ren à chamada de voz do Discord (entra no canal de voz onde o usuário está ou no canal especificado). "
                    "Use quando o usuário pedir 'Ren entra na call', 'entra no Discord', 'conecta na call'."
                ),
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "channel_name": types.Schema(
                            type="STRING",
                            description="Nome opcional do canal de voz para entrar (ex: 'Geral', 'jogos', 'sala').",
                        ),
                    },
                ),
            ),
            types.FunctionDeclaration(
                name="leave_discord_call",
                description=(
                    "Desconecta o Ren de qualquer chamada de voz do Discord. "
                    "Use quando o usuário disser 'Ren sai da call', 'desconecta do Discord', 'sai do canal'."
                ),
                parameters=types.Schema(
                    type="OBJECT",
                    properties={},
                ),
            ),
            types.FunctionDeclaration(
                name="get_discord_status",
                description=(
                    "Verifica se o bot do Ren está conectado a algum canal de voz no Discord e qual o status da conexão."
                ),
                parameters=types.Schema(
                    type="OBJECT",
                    properties={},
                ),
            ),
            types.FunctionDeclaration(
                name="search_obsidian_notes",
                description=(
                    "Pesquisa nas notas Markdown do cofre Obsidian (Argon Hydrogen) usando busca FTS5."
                ),
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "query": types.Schema(
                            type="STRING",
                            description="Termos ou palavras-chave para buscar no Obsidian.",
                        )
                    },
                    required=["query"],
                ),
            ),
            types.FunctionDeclaration(
                name="organize_obsidian_vault",
                description=(
                    "Delega para um modelo de texto Gemini (gemini-2.5-pro) a organização completa do cofre Obsidian: "
                    "estrutura notas, adiciona [[wikilinks]] entre ideias relacionadas e cria MOCs (mapas de conteúdo) "
                    "como um segundo cérebro."
                ),
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "focus": types.Schema(
                            type="STRING",
                            description="Instrução ou foco opcional do usuário (ex: 'conectar projetos', 'organizar tudo').",
                        )
                    },
                ),
            ),
            types.FunctionDeclaration(
                name="close_assistant",
                description=(
                    "Encerra a conversa atual (voltando para espera/standby) ou "
                    "fecha o programa do Ren completamente quando solicitado pelo usuário."
                ),
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "mode": types.Schema(
                            type="STRING",
                            description=(
                                "Use 'standby' para encerrar apenas a conversa atual "
                                "(ex: tchau, até logo, pode descansar) ou 'shutdown' "
                                "para fechar o programa completamente (ex: encerrar sistema, fechar programa)."
                            ),
                        )
                    },
                    required=["mode"],
                ),
            ),
        ]
        for decl in declarations:
            decl.behavior = types.Behavior.NON_BLOCKING
        return declarations

    def _emit_widget(self, widget: str, payload: Dict[str, Any]) -> None:
        if self.gateway is not None:
            self.gateway.emit_ui_widget(widget, payload)

    def execute_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name == "save_memory":
            fact = str(args.get("fact", ""))
            msg = self.save_memory(fact)
            print(f"[Ren] Memória salva: {fact}", flush=True)
            self._emit_widget("memory", {"action": "saved", "fact": fact})
            return {"result": msg}
        if name == "delete_memory":
            q = str(args.get("query", ""))
            msg = self.delete_memory(q)
            print(f"[Ren] Memória atualizada: {msg}", flush=True)
            self._emit_widget("memory", {"action": "deleted", "query": q})
            return {"result": msg}
        if name == "search_local_knowledge":
            query = str(args.get("query", ""))
            print(f"[Ren] Consultando documentos locais: '{query}'", flush=True)
            res = self.search_local_knowledge(query)
            self._emit_widget("knowledge", {"query": query, "result": res[:240]})
            return {"result": res}
        if name == "search_obsidian_notes":
            query = str(args.get("query", ""))
            print(f"[Ren] Buscando no Obsidian: '{query}'", flush=True)
            res = self.search_obsidian_notes(query)
            self._emit_widget("obsidian", {"query": query, "result": res[:240]})
            return {"result": res}
        if name == "organize_obsidian_vault":
            focus = str(args.get("focus", ""))
            print(f"[Ren] Organizando Obsidian com modelo de texto ({focus or 'geral'})...", flush=True)
            res = self.organize_obsidian_vault(focus)
            self._emit_widget("obsidian_organize", {"focus": focus, "result": res[:240]})
            return {"result": res}
        if name == "look_at_screen":
            from src.vision.screen import get_active_window_info

            _, win_title, _, _ = get_active_window_info()
            win_ctx = f" A janela em foco no momento é '{win_title}'." if win_title else ""
            self._emit_widget("vision", {"window": win_title or "Monitor Primário"})
            return {
                "result": (
                    "A captura de tela atual do monitor do usuário acabou de ser "
                    f"enviada como imagem na sessão.{win_ctx} Analise a imagem e responda diretamente."
                )
            }
        if name == "play_music":
            q = str(args.get("query", ""))
            plat = str(args.get("platform", "spotify"))
            print(f"[Ren] Tocando música ({plat}): '{q}'", flush=True)
            res = play_music(q, plat)
            self._emit_widget("music_player", {"query": q, "platform": plat, "status": res})
            return {"result": res}
        if name == "control_media":
            act = str(args.get("action", "play_pause"))
            steps = int(args.get("steps", 5) or 5)
            print(f"[Ren] Controle de mídia: {act}", flush=True)
            res = control_media(act, steps)
            self._emit_widget("media_control", {"action": act, "result": res})
            return {"result": res}
        if name == "open_application":
            app = str(args.get("app_name", ""))
            print(f"[Ren] Abrindo aplicativo: {app}", flush=True)
            res = open_application(app)
            self._emit_widget("app_launcher", {"app": app, "result": res})
            return {"result": res}
        if name == "open_website_or_search":
            target = str(args.get("target", ""))
            sq = str(args.get("search_query", ""))
            print(f"[Ren] Abrindo site: {target} {sq}".strip(), flush=True)
            res = open_website_or_search(target, sq)
            self._emit_widget("browser", {"target": target, "search_query": sq})
            return {"result": res}
        if name == "read_clipboard":
            print("[Ren] Lendo área de transferência (Ctrl+C)...", flush=True)
            return {"result": read_clipboard()}
        if name == "write_clipboard":
            print("[Ren] Copiando texto para a área de transferência (Ctrl+V)...", flush=True)
            return {"result": write_clipboard(str(args.get("text", "")))}
        if name == "get_pc_status":
            print("[Ren] Verificando status de RAM e disco do PC...", flush=True)
            res = get_pc_status()
            self._emit_widget("pc_status", {"summary": res})
            return {"result": res}
        if name == "get_executive_briefing":
            city = str(args.get("city", ""))
            print("[Ren] Montando Briefing Executivo em tempo real...", flush=True)
            res = build_executive_briefing(city=city, memories=self.list_memories())
            self._emit_widget("executive_briefing", {"city": city, "summary": res})
            return {"result": res}
        if name == "read_recent_emails":
            unread = bool(args.get("unread_only", True))
            lim = int(args.get("limit", 4) or 4)
            print("[Ren] Verificando caixa de entrada de e-mails...", flush=True)
            res = read_recent_emails(unread_only=unread, limit=lim)
            self._emit_widget("email_inbox", {"unread_only": unread, "summary": res[:300]})
            return {"result": res}
        if name == "search_emails":
            q = str(args.get("query", ""))
            print(f"[Ren] Pesquisando nos e-mails por: '{q}'", flush=True)
            res = search_emails(q)
            self._emit_widget("email_search", {"query": q, "summary": res[:300]})
            return {"result": res}
        if name == "set_timer":
            secs = int(args.get("duration_seconds", 60) or 60)
            lbl = str(args.get("label", "Timer"))
            msg = self.timers.set_timer(secs, lbl)
            print(f"[Ren] {msg}", flush=True)
            self._emit_widget("timer", {"label": lbl, "duration_seconds": secs, "message": msg})
            return {"result": msg}
        if name == "create_voice_note":
            title = str(args.get("title", "Nota"))
            content = str(args.get("content", ""))
            msg = self.create_voice_note(title, content)
            print(f"[Ren] {msg}", flush=True)
            self._emit_widget("voice_note", {"title": title, "message": msg})
            return {"result": msg}
        if name == "join_discord_call":
            ch = str(args.get("channel_name", "")).strip() or None
            print(f"[Ren] Conectando à call do Discord (canal: {ch or 'automático'})...", flush=True)
            if self.discord_bot is not None:
                res = self.discord_bot.join_call(ch)
            else:
                res = (
                    "O bot do Discord não está ativo. "
                    "Configure o token DISCORD_BOT_TOKEN no arquivo .env para ativar a integração de voz no Discord."
                )
            self._emit_widget("discord", {"action": "join", "result": res})
            return {"result": res}
        if name == "leave_discord_call":
            print("[Ren] Desconectando da chamada do Discord...", flush=True)
            if self.discord_bot is not None:
                res = self.discord_bot.leave_call()
            else:
                res = "O bot do Discord não está ativo no momento."
            self._emit_widget("discord", {"action": "leave", "result": res})
            return {"result": res}
        if name == "get_discord_status":
            print("[Ren] Verificando status do Discord...", flush=True)
            if self.discord_bot is not None:
                res = self.discord_bot.get_status()
            else:
                res = "O bot do Discord não está ativo (DISCORD_BOT_TOKEN vazio no .env)."
            self._emit_widget("discord", {"action": "status", "result": res})
            return {"result": res}
        if name == "close_assistant":
            mode = str(args.get("mode", "standby")).lower()
            return {"result": f"Encerramento agendado ({mode}). Diga uma despedida curta."}
        return {"error": f"Ferramenta desconhecida: {name}"}
