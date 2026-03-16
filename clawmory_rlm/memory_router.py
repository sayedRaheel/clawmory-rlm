from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .structured_memory import StructuredMemoryStore

if TYPE_CHECKING:
    from .chat import MemoryChat


@dataclass
class MemoryResponse:
    mode: str
    content: str


class MemoryRouter:
    def __init__(self, chat: MemoryChat, structured_store: StructuredMemoryStore, namespace: str = "default"):
        self.chat = chat
        self.structured_store = structured_store
        self.namespace = namespace

    def bootstrap(self, goal: str) -> MemoryResponse:
        content = self.structured_store.bootstrap_context(goal, namespace=self.namespace)
        if content:
            return MemoryResponse(mode="bootstrap", content=content)
        return MemoryResponse(mode="bootstrap", content="")

    def recall(self, query: str) -> MemoryResponse:
        mode = self.classify(query)
        if mode == "fast":
            fast = self.structured_store.fast_lookup(query, namespace=self.namespace)
            if fast:
                return MemoryResponse(mode="fast", content=fast)

        try:
            return MemoryResponse(mode="rlm", content=self.chat._rlm_response(query))
        except TimeoutError:
            fast = self.structured_store.fast_lookup(query, namespace=self.namespace)
            if fast:
                return MemoryResponse(
                    mode="fallback",
                    content="Deep recall timed out. Structured fallback:\n" + fast,
                )
            raise

    @staticmethod
    def classify(query: str) -> str:
        q = query.lower()
        if any(phrase in q for phrase in ("last session", "resume", "where did we leave off", "open loop")):
            return "fast"
        if any(phrase in q for phrase in ("preference", "prefer", "decision", "final decision")):
            return "fast"
        if re.search(r"\b(latest|final|remember|what did we decide)\b", q):
            return "fast"
        return "rlm"
