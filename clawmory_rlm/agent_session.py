from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .session_memory import SessionBootstrap, SessionMemoryManager


@dataclass
class AgentSessionState:
    session_id: str
    goal: str
    system_prompt: str
    bootstrap_mode: str


class AgentSession:
    """
    Small runtime wrapper for external agent loops.

    Typical usage:
    1. create the session with a goal
    2. inject `state.system_prompt` into the agent's initial prompt
    3. call `log_user()` / `log_assistant()` as turns happen
    4. call `close()` on shutdown
    """

    def __init__(
        self,
        goal: str,
        session_label: Optional[str] = None,
        history_path: Optional[str] = None,
        structured_memory_path: Optional[str] = None,
        namespace: str = "default",
    ):
        self.manager = SessionMemoryManager(
            history_path=history_path,
            structured_memory_path=structured_memory_path,
            namespace=namespace,
        )
        bootstrap = self.manager.start_session(goal, session_label=session_label)
        self.state = AgentSessionState(
            session_id=bootstrap.session_id,
            goal=goal,
            system_prompt=bootstrap.as_system_prompt(),
            bootstrap_mode=bootstrap.mode,
        )
        self._closed = False

    def initial_messages(self) -> List[dict]:
        return [{"role": "system", "content": self.state.system_prompt}]

    def log_turn(self, role: str, content: str) -> None:
        self.manager.log_turn(role, content)

    def log_user(self, content: str) -> None:
        self.log_turn("user", content)

    def log_assistant(self, content: str) -> None:
        self.log_turn("assistant", content)

    def close(self, label: Optional[str] = None) -> str:
        if self._closed:
            return ""
        self._closed = True
        return self.manager.end_session(label=label)

    def __enter__(self) -> "AgentSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
