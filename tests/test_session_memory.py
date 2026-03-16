from pathlib import Path

from clawmory_rlm.chat import MemoryChat
from clawmory_rlm.session_memory import SessionMemoryManager


def test_start_session_without_prior_history_returns_empty_bootstrap(tmp_path: Path):
    history_path = tmp_path / "history.jsonl"
    structured_path = tmp_path / "structured.json"
    chat = MemoryChat(history_path=history_path, structured_memory_path=structured_path, namespace="default")
    manager = SessionMemoryManager(chat=chat, structured_memory_path=structured_path, namespace="default")

    bootstrap = manager.start_session("continue work")

    assert bootstrap.query == "continue work"
    assert bootstrap.retrieved_context == ""
    assert "No relevant prior-session memory" in bootstrap.as_system_prompt()


def test_start_session_uses_existing_chat_query(tmp_path: Path, monkeypatch):
    history_path = tmp_path / "history.jsonl"
    structured_path = tmp_path / "structured.json"
    chat = MemoryChat(history_path=history_path, structured_memory_path=structured_path, namespace="default")
    chat.add_turn("user", "Use Python.")
    chat.add_turn("assistant", "Noted.")

    def fake_query(prompt: str) -> str:
        assert "Current session goal: resume backend work" in prompt
        return "Prior decision: use Python."

    monkeypatch.setattr(chat, "query", fake_query)
    manager = SessionMemoryManager(chat=chat, structured_memory_path=structured_path, namespace="default")

    bootstrap = manager.start_session("resume backend work", session_label="backend")

    assert bootstrap.retrieved_context == "Prior decision: use Python."
    rows = history_path.read_text().splitlines()
    assert any("SESSION_BOOTSTRAP" in row for row in rows)


def test_end_session_writes_structured_summary(tmp_path: Path):
    history_path = tmp_path / "history.jsonl"
    structured_path = tmp_path / "structured.json"
    chat = MemoryChat(history_path=history_path, structured_memory_path=structured_path, namespace="default")
    manager = SessionMemoryManager(chat=chat, structured_memory_path=structured_path, namespace="default")

    manager.start_session("resume backend work", session_label="backend")
    manager.log_turn("user", "Final decision: use Python with FastAPI.")
    manager.log_turn("assistant", "Next step: wire the startup bootstrap.")
    summary = manager.end_session(label="backend")

    assert "Decisions:" in summary or "Open loops:" in summary
    assert structured_path.exists()
