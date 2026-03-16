from pathlib import Path

from clawmory_rlm.chat import MemoryChat
from clawmory_rlm.session_memory import SessionMemoryManager
from clawmory_rlm.structured_memory import StructuredMemoryBuilder, StructuredMemoryStore


def test_named_namespace_without_memory_does_not_fallback_to_global_query(tmp_path: Path, monkeypatch):
    history_path = tmp_path / "history.jsonl"
    structured_path = tmp_path / "structured.json"
    chat = MemoryChat(history_path=history_path, structured_memory_path=structured_path, namespace="alpha")
    chat.add_turn("user", "Some unrelated global-looking history")

    def fail_query(_prompt: str):
        raise AssertionError("query() should not be called for empty named namespace")

    monkeypatch.setattr(chat, "query", fail_query)
    manager = SessionMemoryManager(chat=chat, structured_memory_path=structured_path, namespace="alpha")

    bootstrap = manager.start_session("resume alpha")
    assert bootstrap.retrieved_context == ""


def test_default_namespace_can_fallback_to_query(tmp_path: Path, monkeypatch):
    history_path = tmp_path / "history.jsonl"
    structured_path = tmp_path / "structured.json"
    chat = MemoryChat(history_path=history_path, structured_memory_path=structured_path, namespace="default")
    chat.add_turn("user", "Use Python.")

    monkeypatch.setattr(chat, "query", lambda prompt: "Recovered from raw history")
    manager = SessionMemoryManager(chat=chat, structured_memory_path=structured_path, namespace="default")

    bootstrap = manager.start_session("resume default")
    assert bootstrap.retrieved_context == "Recovered from raw history"


def test_has_namespace_memory_is_scoped(tmp_path: Path):
    store = StructuredMemoryStore(path=tmp_path / "structured.json")
    builder = StructuredMemoryBuilder()
    turns = [{"turn_index": 0, "role": "user", "content": "Final decision: use FastAPI."}]
    record = builder.build_session_record("alpha", "s1", "alpha", turns)
    builder.update_store(store, record, turns)

    assert store.has_namespace_memory("alpha") is True
    assert store.has_namespace_memory("beta") is False
