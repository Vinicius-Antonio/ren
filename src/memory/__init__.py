"""Módulo D: Memória Persistente e Semântica em SQLite (`src/memory/`) — Hermes 2.0."""

from src.memory.compactor import ContextCompactor, DEFAULT_COMPACT_MODEL, DEFAULT_COMPACT_THRESHOLD
from src.memory.obsidian_organizer import NoteUpdate, ObsidianOrganizer, VaultOrganizationPlan
from src.memory.sqlite_store import ConsolidationPlan, EpisodicMemoryStore, MemoryOperation, VALID_CATEGORIES

__all__ = [
    "ConsolidationPlan",
    "ContextCompactor",
    "DEFAULT_COMPACT_MODEL",
    "DEFAULT_COMPACT_THRESHOLD",
    "EpisodicMemoryStore",
    "MemoryOperation",
    "NoteUpdate",
    "ObsidianOrganizer",
    "VALID_CATEGORIES",
    "VaultOrganizationPlan",
]
