from pathlib import Path

from clawmory_rlm.chat import MemoryChat
from clawmory_rlm.memory_router import MemoryRouter
from clawmory_rlm.structured_memory import StructuredMemoryBuilder, StructuredMemoryStore


def test_router_uses_fast_mode_for_last_session_queries(tmp_path: Path):
    history_path = tmp_path / "history.jsonl"
    structured_path = tmp_path / "structured.json"
    chat = MemoryChat(history_path=history_path, structured_memory_path=structured_path, namespace="default")
    store = StructuredMemoryStore(path=structured_path)
    builder = StructuredMemoryBuilder()
    turns = [
        {"turn_index": 0, "role": "user", "content": "Final decision: use FastAPI."},
        {"turn_index": 1, "role": "assistant", "content": "Need to wire start-session into the runner."},
    ]
    record = builder.build_session_record("default", "s2", "resume", turns)
    builder.update_store(store, record, turns)

    router = MemoryRouter(chat, store, namespace="default")
    response = router.recall("What did we decide last session?")

    assert response.mode == "fast"
    assert "FACT" in response.content or "LATEST SESSION" in response.content
