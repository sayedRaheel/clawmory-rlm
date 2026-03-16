from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from .chat import MemoryChat
from .memory_router import MemoryRouter
from .memory_store import MemoryStore
from .structured_memory import StructuredMemoryBuilder, StructuredMemoryStore


@dataclass
class SessionBootstrap:
    session_id: str
    query: str
    retrieved_context: str
    mode: str = "bootstrap"

    def as_system_prompt(self) -> str:
        if not self.retrieved_context.strip():
            return "No relevant prior-session memory was retrieved."
        return (
            "Relevant memory from previous sessions:\n"
            f"{self.retrieved_context}"
        )


class SessionMemoryManager:
    """
    Thin orchestration layer for cross-session continuity.

    Use `start_session()` at the beginning of a run to pull forward
    relevant historical context, then call `log_turn()` as the session
    progresses to build durable memory for the next run.
    """

    def __init__(
        self,
        chat: Optional[MemoryChat] = None,
        history_path: Optional[str] = None,
        structured_memory_path: Optional[str] = None,
        namespace: str = "default",
    ):
        self.chat = chat or MemoryChat(
            history_path=history_path,
            structured_memory_path=structured_memory_path,
            namespace=namespace,
        )
        self.structured_store = self.chat.structured_store
        self.builder = StructuredMemoryBuilder()
        self.namespace = namespace
        self.router = MemoryRouter(self.chat, self.structured_store, namespace=self.namespace)

    @property
    def store(self) -> MemoryStore:
        return self.chat.store

    def start_session(self, query: str, session_label: Optional[str] = None) -> SessionBootstrap:
        session_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.store.begin_session(session_id=session_id, label=session_label)

        bootstrap_response = self.router.bootstrap(query)
        retrieved_context = bootstrap_response.content
        if (
            not retrieved_context
            and (
                self.namespace == "default"
                or self.structured_store.has_namespace_memory(self.namespace)
            )
            and self.store.history_turn_count(exclude_session_markers=True) > 0
        ):
            retrieved_context = self.chat.query(
                "You are retrieving durable memory for a new session. "
                "Return the most relevant facts, prior decisions, constraints, "
                "and unfinished work needed to continue this thread.\n\n"
                f"Current session goal: {query}"
            )

        bootstrap = SessionBootstrap(
            session_id=session_id,
            query=query,
            retrieved_context=retrieved_context,
            mode="bootstrap" if bootstrap_response.content else "rlm",
        )

        self.store.add(
            "system",
            (
                f"[SESSION_BOOTSTRAP][namespace={self.namespace}] Goal: {query}\n"
                f"Retrieved memory:\n{bootstrap.as_system_prompt()}"
            ),
        )
        return bootstrap

    def log_turn(self, role: str, content: str) -> None:
        self.chat.add_turn(role, content)

    def end_session(self, label: Optional[str] = None) -> str:
        session_id = self.store.current_session_id() or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        turns = self.store.current_session_turns()
        session_record = self.builder.build_session_record(
            namespace=self.namespace,
            session_id=session_id,
            label=label or "",
            turns=turns,
        )
        self.builder.update_store(self.structured_store, session_record, turns)
        return session_record.summary
