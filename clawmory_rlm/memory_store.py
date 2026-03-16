"""
memory_store.py

Append-only conversation history store.
Stores every turn (user + assistant) with metadata.
Used as the source-of-truth that the MemoryRLM searches over.
"""

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional

from .config import resolve_history_path


@dataclass
class Turn:
    role: str           # "user" or "assistant"
    content: str
    timestamp: float
    turn_index: int


class MemoryStore:
    """
    Append-only store for conversation turns.

    Provides serialization helpers so MemoryRLM can expose
    the history as both a plain string and a structured list
    inside the REPL environment.
    """

    def __init__(self, history_path: Optional[str] = None, autoload: bool = False):
        self.turns: List[Turn] = []
        self.history_path = resolve_history_path(history_path)
        if autoload:
            self.load()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(self, role: str, content: str) -> None:
        """Append a new turn to the history."""
        turn = Turn(
            role=role,
            content=content,
            timestamp=time.time(),
            turn_index=len(self.turns),
        )
        self.turns.append(turn)
        self._append_to_disk(turn)

    def begin_session(self, session_id: Optional[str] = None, label: Optional[str] = None) -> None:
        session_id = session_id or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        suffix = f" | {label}" if label else ""
        self.add("system", f"--- Session {session_id}{suffix} ---")

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_string(self) -> str:
        """
        Flat text representation exposed as `context` in the REPL.

        Format per line:
            [Turn 0][USER]: hello
            [Turn 1][ASSISTANT]: hi there
        """
        lines = []
        for t in self.turns:
            lines.append(f"[Turn {t.turn_index}][{t.role.upper()}]: {t.content}")
        return "\n".join(lines)

    def to_dicts(self) -> List[Dict]:
        """
        Structured representation exposed as `history_turns` in the REPL.
        Each element: {"turn_index", "role", "content", "timestamp"}
        """
        return [
            {
                "turn_index": t.turn_index,
                "role": t.role,
                "content": t.content,
                "timestamp": t.timestamp,
            }
            for t in self.turns
        ]

    def load(self) -> None:
        """Load turns from the JSONL history file if present."""
        self.turns = []
        if not self.history_path.exists():
            return

        with self.history_path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                self.turns.append(
                    Turn(
                        role=row.get("role", "user"),
                        content=row.get("content", ""),
                        timestamp=self._parse_timestamp(row.get("timestamp", time.time())),
                        turn_index=int(row.get("turn_index", idx)),
                    )
                )

    def history_turn_count(self, exclude_session_markers: bool = False) -> int:
        if not exclude_session_markers:
            return len(self.turns)
        return sum(1 for turn in self.turns if not self._is_session_marker(turn.content))

    def to_openai_messages(self) -> List[Dict[str, str]]:
        """Return history in OpenAI messages format (for normal-mode chat)."""
        return [{"role": t.role, "content": t.content} for t in self.turns]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def total_chars(self) -> int:
        return sum(len(t.content) for t in self.turns)

    def current_session_id(self) -> Optional[str]:
        for turn in reversed(self.turns):
            if turn.role == "system" and self._is_session_marker(turn.content):
                marker = turn.content.replace("--- Session ", "").replace("---", "").strip()
                return marker.split("|")[0].strip()
        return None

    def current_session_turns(self) -> List[Dict]:
        session_start = -1
        for idx in range(len(self.turns) - 1, -1, -1):
            turn = self.turns[idx]
            if turn.role == "system" and self._is_session_marker(turn.content):
                session_start = idx + 1
                break
        selected = self.turns[session_start:] if session_start >= 0 else self.turns
        return [
            {
                "turn_index": t.turn_index,
                "role": t.role,
                "content": t.content,
                "timestamp": t.timestamp,
            }
            for t in selected
        ]

    def total_turns(self) -> int:
        return len(self.turns)

    def __len__(self) -> int:
        return len(self.turns)

    def __repr__(self) -> str:
        return f"MemoryStore({len(self.turns)} turns, {self.total_chars():,} chars)"

    def _append_to_disk(self, turn: Turn) -> None:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "turn_index": turn.turn_index,
            "timestamp": turn.timestamp,
            "role": turn.role,
            "content": turn.content,
        }
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    @staticmethod
    def _is_session_marker(content: str) -> bool:
        return content.startswith("--- Session ")

    @staticmethod
    def _parse_timestamp(value) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                try:
                    normalized = value.replace("Z", "+00:00")
                    return datetime.fromisoformat(normalized).timestamp()
                except ValueError:
                    return time.time()
        return time.time()
