import json
from pathlib import Path

from clawmory_rlm.memory_store import MemoryStore


def test_memory_store_persists_and_loads(tmp_path: Path):
    history_path = tmp_path / "history.jsonl"
    store = MemoryStore(history_path=history_path)
    store.add("user", "hello")
    store.add("assistant", "hi")

    reloaded = MemoryStore(history_path=history_path, autoload=True)
    assert reloaded.total_turns() == 2
    assert reloaded.turns[0].content == "hello"
    assert reloaded.turns[1].content == "hi"


def test_begin_session_writes_system_marker(tmp_path: Path):
    history_path = tmp_path / "history.jsonl"
    store = MemoryStore(history_path=history_path)
    store.begin_session(session_id="session-1", label="resume-work")

    rows = [json.loads(line) for line in history_path.read_text().splitlines()]
    assert rows[0]["role"] == "system"
    assert rows[0]["content"] == "--- Session session-1 | resume-work ---"
