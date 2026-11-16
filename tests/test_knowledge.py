from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List

import pytest

from src.audio.stream_manager import AudioStreamManager
from src.gemini.knowledge import KnowledgeManager
from src.gemini.prompts import build_system_instruction, is_farewell, is_shutdown_command
from src.gemini.session import GeminiLiveSession
from tests.conftest import FakeGenaiClient, FakePyAudioModule, make_settings


def test_knowledge_manager_save_list_and_delete_memories(tmp_path: Path) -> None:
    mem_file = tmp_path / "data" / "memory.json"
    km = KnowledgeManager(memory_file=mem_file, knowledge_dir=tmp_path / "knowledge")

    assert km.list_memories() == []
    km.save_memory("O usuário se chama Vinícius")
    km.save_memory("O usuário se chama Vinícius")  # duplicata ignorada
    km.save_memory("Mora em São Paulo")
    assert km.list_memories() == [
        "O usuário se chama Vinícius",
        "Mora em São Paulo",
    ]

    km.delete_memory("são paulo")
    assert km.list_memories() == ["O usuário se chama Vinícius"]


def test_knowledge_manager_searches_local_documents(tmp_path: Path) -> None:
    k_dir = tmp_path / "knowledge"
    k_dir.mkdir()
    (k_dir / "README.md").write_text("ignorar readme", encoding="utf-8")
    (k_dir / "projeto_hermes.md").write_text(
        "O arquitetura do Hermes usa PCM 16kHz na entrada e 24kHz na saída.",
        encoding="utf-8",
    )
    km = KnowledgeManager(memory_file=tmp_path / "mem.json", knowledge_dir=k_dir)

    assert km.list_knowledge_files() == ["projeto_hermes.md"]
    res = km.search_local_knowledge("arquitetura PCM")
    assert "projeto_hermes.md" in res
    assert "24kHz" in res


def test_build_system_instruction_injects_date_memories_and_files() -> None:
    fixed_dt = datetime(2026, 11, 7, 20, 50)
    prompt = build_system_instruction(
        memories=["Prefere respostas curtas"],
        knowledge_files=["notas.md"],
        now=fixed_dt,
    )
    assert "sábado, 7 de novembro de 2026" in prompt
    assert "20:50" in prompt
    assert "Prefere respostas curtas" in prompt
    assert "notas.md" in prompt


def test_shutdown_commands_and_farewell_phrases() -> None:
    assert is_shutdown_command("Hermes, encerrar sistema agora")
    assert is_shutdown_command("Pode fechar o programa")
    assert is_shutdown_command("Desligar completamente")
    assert is_farewell("Encerrar sistema")
    assert not is_shutdown_command("Tchau, até logo")
    assert is_farewell("Tchau, até logo")


def test_sliding_window_and_rolling_summary_stay_bounded(tmp_path: Path) -> None:
    from datetime import timedelta
    from src.gemini.knowledge import MAX_SUMMARY_CHARS

    km = KnowledgeManager(
        memory_file=tmp_path / "memory.json", knowledge_dir=tmp_path / "knowledge"
    )
    t0 = datetime(2026, 11, 7, 14, 0)

    # Registra 12 turnos: apenas os 4 últimos ficam na janela literal; os 8 anteriores
    # são comprimidos no resumo acumulativo respeitando o teto de caracteres.
    for i in range(1, 13):
        km.record_turn(
            f"Pergunta número {i} sobre o projeto",
            f"Resposta número {i} explicando o detalhe técnico",
            now=t0 + timedelta(minutes=i),
        )

    store = km._load_store()
    assert len(store["recent_turns"]) == 4
    assert len(store["rolling_summary"]) <= MAX_SUMMARY_CHARS
    assert "Pergunta número 12" in store["recent_turns"][-1]["user"]

    ctx = km.get_conversation_context(now=t0 + timedelta(minutes=20))
    assert "Resumo anterior:" in ctx
    assert "Últimas falas recentes:" in ctx
    assert "Pergunta número 12" in ctx
    assert len(ctx) < 1300  # Nunca fica grande!

    # Após 7 horas (> 6h), os turnos literais antigos também são dobrados no resumo
    ctx_later = km.get_conversation_context(now=t0 + timedelta(hours=7))
    assert "Resumo anterior:" in ctx_later
    assert "Últimas falas recentes:" not in ctx_later


def test_call_history_keeps_discord_author(tmp_path: Path) -> None:
    km = KnowledgeManager(memory_file=tmp_path / "memory.json", knowledge_dir=tmp_path / "knowledge")
    km.record_turn("Qual é a hora?", "São duas horas.", speaker_id=202, speaker_name="Bruno")
    turn = km._load_store()["recent_turns"][-1]
    assert turn["speaker_id"] == "202"
    assert turn["speaker_name"] == "Bruno"
    assert "Bruno (Discord 202)" in km.get_conversation_context()


def test_session_logs_google_search_queries_and_handles_tool_calls(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    km = KnowledgeManager(
        memory_file=tmp_path / "memory.json", knowledge_dir=tmp_path / "knowledge"
    )
    sess = GeminiLiveSession(
        make_settings(),
        AudioStreamManager(FakePyAudioModule()),
        FakeGenaiClient(),
        knowledge_manager=km,
    )

    msg = SimpleNamespace(
        go_away=None,
        server_content=SimpleNamespace(
            interrupted=False,
            grounding_metadata=SimpleNamespace(
                web_search_queries=["cotação dólar hoje", "clima são paulo"]
            ),
            input_transcription=None,
            model_turn=None,
            turn_complete=True,
        ),
    )
    sess._handle_message(msg)
    out = capsys.readouterr().out
    assert "Pesquisando no Google: 'cotação dólar hoje', 'clima são paulo'" in out

    tool_responses: List[Any] = []

    class FakeSessionWithTools:
        async def send_tool_response(self, *, function_responses: Any) -> None:
            tool_responses.extend(function_responses)

    tool_call = SimpleNamespace(
        function_calls=[
            SimpleNamespace(
                id="call-1",
                name="save_memory",
                args={"fact": "Gosta de programação em Python"},
            )
        ]
    )
    asyncio.run(sess._handle_tool_call(FakeSessionWithTools(), tool_call))
    assert km.list_memories() == ["Gosta de programação em Python"]
    assert len(tool_responses) == 1
    assert tool_responses[0].name == "save_memory"


def test_all_function_declarations_are_non_blocking(tmp_path: Path) -> None:
    from google.genai import types

    km = KnowledgeManager(
        memory_file=tmp_path / "memory.json", knowledge_dir=tmp_path / "knowledge"
    )
    decls = km.build_function_declarations()
    assert len(decls) >= 18
    names = {d.name for d in decls}
    assert "search_obsidian_notes" in names
    for decl in decls:
        assert decl.behavior == types.Behavior.NON_BLOCKING, f"{decl.name} não é NON_BLOCKING"


def test_obsidian_fts5_search_and_voice_note_frontmatter(tmp_path: Path) -> None:
    vault = tmp_path / "ArgonVault"
    (vault / "Projetos").mkdir(parents=True)
    (vault / "Projetos" / "Argon.md").write_text(
        "# Argon Hydrogen\nArquitetura de rede neural e pipeline de voz baixa latência.",
        encoding="utf-8",
    )
    km = KnowledgeManager(
        memory_file=tmp_path / "memory.json",
        knowledge_dir=tmp_path / "knowledge",
        obsidian_vault_path=vault,
    )

    res = km.search_obsidian_notes("latência")
    assert "Argon Hydrogen" in res
    assert "Projetos/Argon.md" in res

    note_msg = km.execute_tool(
        "create_voice_note",
        {"title": "Ideia Pipeline", "content": "Testar WebSocket com resumption handle."},
    )
    assert "VoiceNotes/ideia_pipeline.md" in note_msg["result"]
    note_file = vault / "VoiceNotes" / "ideia_pipeline.md"
    assert note_file.is_file()
    note_text = note_file.read_text(encoding="utf-8")
    assert note_text.startswith("---\n")
    assert "voice-note" in note_text

    # Busca FTS5 deve encontrar a nota recém-criada via triggers SQL
    hit = km.search_obsidian_notes("resumption")
    assert "Ideia Pipeline" in hit


def test_compactor_compacts_15_turns_and_injects_silently(tmp_path: Path) -> None:
    from src.memory.compactor import ContextCompactor

    km = KnowledgeManager(
        memory_file=tmp_path / "memory.json", knowledge_dir=tmp_path / "knowledge"
    )
    for i in range(1, 16):
        km.record_turn(f"Pergunta {i} sobre bugs", f"Resposta {i} com humor")

    assert len(km.sqlite_store.get_uncompacted_turns()) == 15

    used_models: List[str] = []
    injected_calls: List[Any] = []

    class FakeModels:
        def generate_content(self, *, model: str, contents: str, **_kwargs: Any) -> Any:
            used_models.append(model)
            return SimpleNamespace(text="Resumo compacto dos 15 turnos sobre bugs e humor.")

    class FakeLive:
        async def send_client_content(self, *, turns: Any, turn_complete: bool) -> None:
            injected_calls.append((turns, turn_complete))

    compactor = ContextCompactor(threshold=15)
    summary = asyncio.run(
        compactor.maybe_compact_and_inject(
            store=km.sqlite_store,
            genai_client=SimpleNamespace(models=FakeModels()),
            live_session=FakeLive(),
        )
    )

    assert summary == "Resumo compacto dos 15 turnos sobre bugs e humor."
    assert used_models == ["gemini-2.5-flash"]
    assert km.sqlite_store.get_uncompacted_turns() == []
    assert km.sqlite_store.get_latest_rolling_summary() == summary
    assert len(injected_calls) == 1
    assert injected_calls[0][1] is False


def test_post_session_consolidation_pydantic_add_update_delete_and_voicelog(
    tmp_path: Path,
) -> None:
    from src.memory.sqlite_store import ConsolidationPlan, MemoryOperation

    vault = tmp_path / "ArgonVault"
    vault.mkdir()
    km = KnowledgeManager(
        memory_file=tmp_path / "memory.json",
        knowledge_dir=tmp_path / "knowledge",
        obsidian_vault_path=vault,
    )
    id_keep = km.sqlite_store.add_episodic_memory("Trabalha no Hermes v1", "projeto")
    id_del = km.sqlite_store.add_episodic_memory("Usa tema claro no editor", "preferencia")
    assert id_keep is not None and id_del is not None

    km.record_turn("Agora estou no Hermes v2 e prefiro tema escuro", "Anotado!")

    plan = ConsolidationPlan(
        operations=[
            MemoryOperation(
                op="UPDATE",
                id=id_keep,
                categoria="projeto",
                conteudo="Trabalha no Hermes v2",
            ),
            MemoryOperation(op="DELETE", id=id_del),
            MemoryOperation(
                op="ADD",
                categoria="preferencia",
                conteudo="Prefere tema escuro no editor",
            ),
            # Duplicata proposital: não deve duplicar no SQLite
            MemoryOperation(
                op="ADD",
                categoria="preferencia",
                conteudo="prefere tema escuro no editor",
            ),
        ],
        session_summary="Migração para Hermes v2 e preferência por tema escuro.",
        topics=["Hermes", "Obsidian"],
    )

    class FakeModels:
        def generate_content(self, **_kwargs: Any) -> Any:
            return SimpleNamespace(parsed=plan, text=plan.model_dump_json())

    km.consolidate_post_session(genai_client=SimpleNamespace(models=FakeModels()))

    memories = km.sqlite_store.list_episodic_memories(limit=20)
    contents = [m["conteudo"] for m in memories]
    assert "Trabalha no Hermes v2" in contents
    assert "Prefere tema escuro no editor" in contents
    assert "Usa tema claro no editor" not in contents
    assert len(contents) == 2

    today = datetime.now().strftime("%Y-%m-%d")
    voice_log = vault / "VoiceLogs" / f"{today} - Resumo.md"
    assert voice_log.is_file()
    log_text = voice_log.read_text(encoding="utf-8")
    assert "[[Hermes]]" in log_text
    assert "[[Obsidian]]" in log_text
    assert "Migração para Hermes v2" in log_text


def test_organize_obsidian_vault_creates_wikilinks_and_reindexes(tmp_path: Path) -> None:
    from src.memory.obsidian_organizer import NoteUpdate, VaultOrganizationPlan

    vault = tmp_path / "ArgonVault"
    (vault / "VoiceNotes").mkdir(parents=True)
    (vault / "VoiceNotes" / "ideia_ws.md").write_text(
        "# Ideia WebSocket\nUsar resumption handle na Live API.",
        encoding="utf-8",
    )
    (vault / "VoiceNotes" / "hermes.md").write_text(
        "# Projeto Hermes\nAssistente de voz com Discord.",
        encoding="utf-8",
    )

    plan = VaultOrganizationPlan(
        summary="Conectei o Projeto Hermes com a Ideia WebSocket e criei o Mapa Mental.",
        notes=[
            NoteUpdate(
                path="00 - Mapa Mental.md",
                title="Mapa Mental",
                tags=["moc", "cerebro"],
                content_markdown="# Mapa Mental\n- [[Projeto Hermes]]\n- [[Ideia WebSocket]]",
                links_added=["Projeto Hermes", "Ideia WebSocket"],
            ),
            NoteUpdate(
                path="../fora_do_cofre.md",
                title="Invasor",
                tags=[],
                content_markdown="Não pode ser gravado fora do vault.",
            ),
        ],
    )

    used_models: List[str] = []

    class FakeModels:
        def generate_content(self, *, model: str, **_kwargs: Any) -> Any:
            used_models.append(model)
            return SimpleNamespace(parsed=plan, text=plan.model_dump_json())

    km = KnowledgeManager(
        memory_file=tmp_path / "memory.json",
        knowledge_dir=tmp_path / "knowledge",
        obsidian_vault_path=vault,
        genai_client=SimpleNamespace(models=FakeModels()),
    )

    result = km.execute_tool("organize_obsidian_vault", {"focus": "conectar tudo"})
    assert used_models == ["gemini-2.5-pro"]
    assert "00 - Mapa Mental.md" in result["result"]
    assert not (tmp_path / "fora_do_cofre.md").exists()

    moc_file = vault / "00 - Mapa Mental.md"
    assert moc_file.is_file()
    moc_text = moc_file.read_text(encoding="utf-8")
    assert "[[Projeto Hermes]]" in moc_text
    assert "[[Ideia WebSocket]]" in moc_text

    # Já deve estar indexado no FTS5 imediatamente
    search_hit = km.search_obsidian_notes("Mapa Mental")
    assert "Mapa Mental" in search_hit

