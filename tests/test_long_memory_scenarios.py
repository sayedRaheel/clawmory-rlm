from pathlib import Path

from clawmory_rlm import AgentSession
from clawmory_rlm.structured_memory import StructuredMemoryStore


def test_multi_session_latest_truth_wins_across_long_history(tmp_path: Path):
    history = tmp_path / "history.jsonl"
    structured = tmp_path / "structured.json"

    sessions = [
        (
            "s1",
            [
                ("user", "We first considered Go for the backend."),
                ("assistant", "Noted."),
            ],
        ),
        (
            "s2",
            [
                ("user", "We then moved toward Python."),
                ("assistant", "Noted."),
            ],
        ),
        (
            "s3",
            [
                ("user", "Final decision: Python with FastAPI."),
                ("assistant", "Next step: connect the runner."),
            ],
        ),
        (
            "s4",
            [
                ("user", "We also decided the memory system should stay namespace-aware."),
                ("assistant", "Open loop: add chronology benchmarks."),
            ],
        ),
    ]

    for label, turns in sessions:
        with AgentSession(
            goal=f"Run {label}",
            history_path=history,
            structured_memory_path=structured,
            session_label=label,
            namespace="backend",
        ) as session:
            for role, content in turns:
                session.log_turn(role, content)

    with AgentSession(
        goal="Resume backend work",
        history_path=history,
        structured_memory_path=structured,
        session_label="resume",
        namespace="backend",
    ) as resumed:
        prompt = resumed.state.system_prompt

    assert "Python with FastAPI" in prompt
    assert "connect the runner" in prompt
    assert "namespace-aware" in prompt or "chronology benchmarks" in prompt


def test_project_namespaces_remain_isolated_over_many_sessions(tmp_path: Path):
    history = tmp_path / "history.jsonl"
    structured = tmp_path / "structured.json"

    for namespace, final_decision in (
        ("dispatch-agent", "Final decision: use LiveKit for dispatch-agent."),
        ("clawmory-rlm", "Final decision: use structured memory with RLM fallback."),
    ):
        for idx in range(3):
            with AgentSession(
                goal=f"Session {idx}",
                history_path=history,
                structured_memory_path=structured,
                session_label=f"{namespace}-{idx}",
                namespace=namespace,
            ) as session:
                if idx == 2:
                    session.log_user(final_decision)
                else:
                    session.log_user(f"Progress update {idx} for {namespace}.")

    with AgentSession(
        goal="Resume dispatch-agent",
        history_path=history,
        structured_memory_path=structured,
        session_label="dispatch-resume",
        namespace="dispatch-agent",
    ) as dispatch:
        dispatch_prompt = dispatch.state.system_prompt

    with AgentSession(
        goal="Resume clawmory",
        history_path=history,
        structured_memory_path=structured,
        session_label="clawmory-resume",
        namespace="clawmory-rlm",
    ) as clawmory:
        clawmory_prompt = clawmory.state.system_prompt

    assert "LiveKit" in dispatch_prompt
    assert "structured memory with RLM fallback" not in dispatch_prompt
    assert "structured memory with RLM fallback" in clawmory_prompt
    assert "LiveKit" not in clawmory_prompt


def test_structured_store_retains_only_latest_decision_per_namespace(tmp_path: Path):
    history = tmp_path / "history.jsonl"
    structured = tmp_path / "structured.json"

    with AgentSession(
        goal="Old decision",
        history_path=history,
        structured_memory_path=structured,
        session_label="old",
        namespace="backend",
    ) as session:
        session.log_user("Final decision: use Go.")

    with AgentSession(
        goal="New decision",
        history_path=history,
        structured_memory_path=structured,
        session_label="new",
        namespace="backend",
    ) as session:
        session.log_user("Final decision: use Python with FastAPI.")

    store = StructuredMemoryStore(path=structured)
    facts = [fact for fact in store.data["facts"] if fact.get("namespace") == "backend"]
    final_decisions = [fact for fact in facts if fact["key"] == "final_decision"]

    assert len(final_decisions) == 1
    assert final_decisions[0]["value"] == "use Python with FastAPI."
