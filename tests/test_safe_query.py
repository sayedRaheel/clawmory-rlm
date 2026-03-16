from pathlib import Path

from clawmory_rlm.chat import MemoryChat


def test_safe_query_falls_back_to_structured_memory_on_timeout(tmp_path: Path, monkeypatch):
    history_path = tmp_path / "history.jsonl"
    structured_path = tmp_path / "structured.json"
    chat = MemoryChat(history_path=history_path, structured_memory_path=structured_path, namespace="default")
    chat.structured_store.data["facts"] = [
        {
            "namespace": "default",
            "key": "final_decision",
            "value": "use FastAPI",
            "category": "decision",
            "source_session_id": "s1",
            "source_turn_index": 0,
            "confidence": 0.9,
            "updated_at": "now",
        }
    ]

    def raise_timeout(query: str):
        raise TimeoutError("timed out")

    monkeypatch.setattr(chat, "query", raise_timeout)

    result = chat.safe_query("What was the final backend decision?")
    assert "timed out" in result.lower()
    assert "use FastAPI" in result


def test_safe_query_falls_back_to_recent_history_when_no_structured_memory(tmp_path: Path, monkeypatch):
    history_path = tmp_path / "history.jsonl"
    structured_path = tmp_path / "structured.json"
    chat = MemoryChat(history_path=history_path, structured_memory_path=structured_path, namespace="default")
    chat.add_turn("user", "We first considered Go.")
    chat.add_turn("assistant", "Noted.")
    chat.add_turn("user", "Final decision: use Python with FastAPI.")

    monkeypatch.setattr(chat, "query", lambda _query: (_ for _ in ()).throw(TimeoutError("timed out")))

    result = chat.safe_query("What was the final backend decision?")
    assert "Recent-history fallback" in result
    assert "Python with FastAPI" in result
