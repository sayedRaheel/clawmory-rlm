from pathlib import Path

from clawmory_rlm.structured_memory import StructuredMemoryBuilder, StructuredMemoryStore


def test_structured_memory_extracts_decisions_preferences_and_loops(tmp_path: Path):
    store = StructuredMemoryStore(path=tmp_path / "structured.json")
    builder = StructuredMemoryBuilder()
    turns = [
        {"turn_index": 0, "role": "user", "content": "We decided to use Python for the backend."},
        {"turn_index": 1, "role": "user", "content": "I prefer short answers."},
        {"turn_index": 2, "role": "assistant", "content": "Next step: implement session bootstrap."},
    ]

    record = builder.build_session_record("default", "s1", "test", turns)
    builder.update_store(store, record, turns)

    assert "We decided to use Python for the backend." in record.decisions
    assert "I prefer short answers." in record.preferences
    assert "Next step: implement session bootstrap." in record.open_loops
    assert store.data["facts"]
    assert store.active_open_loops()


def test_bootstrap_context_prioritizes_latest_summary_and_open_loops(tmp_path: Path):
    store = StructuredMemoryStore(path=tmp_path / "structured.json")
    builder = StructuredMemoryBuilder()
    turns = [
        {"turn_index": 0, "role": "user", "content": "Final decision: use FastAPI."},
        {"turn_index": 1, "role": "assistant", "content": "Need to wire start-session into the runner."},
    ]
    record = builder.build_session_record("default", "s2", "resume", turns)
    builder.update_store(store, record, turns)

    context = store.bootstrap_context("Resume the memory system", namespace="default")
    assert "Current goal: Resume the memory system" in context
    assert "Last session summary:" in context
    assert "Open loops:" in context


def test_namespaces_do_not_contaminate_each_other(tmp_path: Path):
    store = StructuredMemoryStore(path=tmp_path / "structured.json")
    builder = StructuredMemoryBuilder()

    alpha_turns = [{"turn_index": 0, "role": "user", "content": "Final decision: use FastAPI."}]
    beta_turns = [{"turn_index": 0, "role": "user", "content": "Final decision: use Django."}]

    builder.update_store(store, builder.build_session_record("alpha", "s1", "alpha", alpha_turns), alpha_turns)
    builder.update_store(store, builder.build_session_record("beta", "s2", "beta", beta_turns), beta_turns)

    alpha_context = store.bootstrap_context("Resume alpha", namespace="alpha")
    beta_context = store.bootstrap_context("Resume beta", namespace="beta")

    assert "FastAPI" in alpha_context
    assert "Django" not in alpha_context
    assert "Django" in beta_context
    assert "FastAPI" not in beta_context


def test_newer_fact_replaces_older_fact_within_same_namespace(tmp_path: Path):
    store = StructuredMemoryStore(path=tmp_path / "structured.json")
    builder = StructuredMemoryBuilder()

    old_turns = [{"turn_index": 0, "role": "user", "content": "Final decision: use Go."}]
    new_turns = [{"turn_index": 1, "role": "user", "content": "Final decision: use Python with FastAPI."}]

    builder.update_store(store, builder.build_session_record("default", "s1", "old", old_turns), old_turns)
    builder.update_store(store, builder.build_session_record("default", "s2", "new", new_turns), new_turns)

    context = store.bootstrap_context("Resume backend work", namespace="default")
    assert "Python with FastAPI" in context
    assert "Final decision: use Go." not in context
