from pathlib import Path

from clawmory_rlm.agent_session import AgentSession


def test_agent_session_resume_and_close(tmp_path: Path):
    history_path = tmp_path / "history.jsonl"
    structured_path = tmp_path / "structured.json"

    with AgentSession(
        goal="Resume work on memory",
        history_path=history_path,
        structured_memory_path=structured_path,
        session_label="memory",
        namespace="alpha",
    ) as session:
        assert session.state.system_prompt
        session.log_user("Final decision: keep structured memory.")
        session.log_assistant("Next step: connect it to the runner.")

    assert structured_path.exists()

    resumed = AgentSession(
        goal="Resume work on memory",
        history_path=history_path,
        structured_memory_path=structured_path,
        session_label="memory",
        namespace="alpha",
    )
    assert "structured memory" in resumed.state.system_prompt.lower()
    resumed.close(label="memory")
