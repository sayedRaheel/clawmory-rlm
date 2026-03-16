import os

import pytest

from clawmory_rlm.chat import MemoryChat


@pytest.mark.skipif(
    not (os.getenv("CLAWMORY_RUN_LIVE_TESTS") == "1" and os.getenv("OPENAI_API_KEY")),
    reason="Live OpenAI test disabled or OPENAI_API_KEY missing.",
)
def test_openai_provider_safe_query_live(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAWMORY_PROVIDER", "openai")
    monkeypatch.setenv("CLAWMORY_MODEL", os.getenv("CLAWMORY_MODEL", "gpt-4o-mini"))

    history_path = tmp_path / "history.jsonl"
    structured_path = tmp_path / "structured.json"
    chat = MemoryChat(history_path=history_path, structured_memory_path=structured_path, namespace="default")
    chat.add_turn("user", "Final decision: use Python with FastAPI.")

    result = chat.safe_query("What was the final backend decision?")
    assert result


@pytest.mark.skipif(
    not (os.getenv("CLAWMORY_RUN_LIVE_TESTS") == "1" and os.getenv("CLAWMORY_PROVIDER", "").lower() == "ollama"),
    reason="Live Ollama test disabled or provider is not ollama.",
)
def test_ollama_provider_safe_query_live(tmp_path):
    history_path = tmp_path / "history.jsonl"
    structured_path = tmp_path / "structured.json"
    chat = MemoryChat(history_path=history_path, structured_memory_path=structured_path, namespace="default")
    chat.add_turn("user", "Final decision: use Python with FastAPI.")

    result = chat.safe_query("What was the final backend decision?")
    assert result
