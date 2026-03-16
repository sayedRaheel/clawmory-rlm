from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import PROJECT_ROOT


DEFAULT_STRUCTURED_MEMORY_PATH = PROJECT_ROOT / "structured_memory.json"

DECISION_PREFIXES = (
    "we decided",
    "final decision",
    "decision",
    "we will",
    "we're going to",
    "we are going to",
)
PREFERENCE_PATTERNS = (
    re.compile(r"\bI prefer\b", re.IGNORECASE),
    re.compile(r"\buser prefers\b", re.IGNORECASE),
    re.compile(r"\bI like\b", re.IGNORECASE),
)
TASK_PATTERNS = (
    re.compile(r"\b(todo|to do|next step|follow up|need to|must|should)\b", re.IGNORECASE),
    re.compile(r"\bopen loop\b", re.IGNORECASE),
    re.compile(r"\bunresolved\b", re.IGNORECASE),
)
RESOLVED_PATTERNS = (
    re.compile(r"\b(done|resolved|completed|fixed|closed)\b", re.IGNORECASE),
)


@dataclass
class FactRecord:
    namespace: str
    key: str
    value: str
    category: str
    source_session_id: str
    source_turn_index: int
    confidence: float = 0.6
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class OpenLoopRecord:
    namespace: str
    id: str
    text: str
    status: str
    source_session_id: str
    source_turn_index: int
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class SessionSummaryRecord:
    namespace: str
    session_id: str
    label: str
    started_at: str
    ended_at: str
    summary: str
    decisions: List[str]
    preferences: List[str]
    open_loops: List[str]


class StructuredMemoryStore:
    def __init__(self, path: Optional[str] = None):
        self.path = Path(path).expanduser().resolve() if path else DEFAULT_STRUCTURED_MEMORY_PATH
        self.data: Dict[str, Any] = {
            "sessions": [],
            "facts": [],
            "open_loops": [],
            "updated_at": None,
        }
        self.load()

    def load(self) -> None:
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))

    def save(self) -> None:
        self.data["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def append_session(self, record: SessionSummaryRecord) -> None:
        self.data["sessions"].append(record.__dict__)
        self.save()

    def upsert_fact(self, record: FactRecord) -> None:
        facts = self.data["facts"]
        for idx, current in enumerate(facts):
            if current["key"] == record.key and current.get("namespace", "default") == record.namespace:
                facts[idx] = record.__dict__
                self.save()
                return
        facts.append(record.__dict__)
        self.save()

    def upsert_open_loop(self, record: OpenLoopRecord) -> None:
        loops = self.data["open_loops"]
        for idx, current in enumerate(loops):
            if current["id"] == record.id and current.get("namespace", "default") == record.namespace:
                loops[idx] = record.__dict__
                self.save()
                return
        loops.append(record.__dict__)
        self.save()

    def active_open_loops(self, namespace: str = "default") -> List[Dict[str, Any]]:
        return [
            loop for loop in self.data["open_loops"]
            if loop["status"] != "resolved" and loop.get("namespace", "default") == namespace
        ]

    def latest_session(self, namespace: str = "default") -> Optional[Dict[str, Any]]:
        sessions = [
            session for session in self.data["sessions"]
            if session.get("namespace", "default") == namespace
        ]
        if not sessions:
            return None
        return sessions[-1]

    def has_namespace_memory(self, namespace: str = "default") -> bool:
        return bool(
            self.latest_session(namespace=namespace)
            or any(fact.get("namespace", "default") == namespace for fact in self.data["facts"])
            or any(loop.get("namespace", "default") == namespace for loop in self.data["open_loops"])
        )

    def bootstrap_context(self, goal: str, namespace: str = "default", limit: int = 6) -> str:
        lines: List[str] = [f"Current goal: {goal}"]
        lines.append(f"Namespace: {namespace}")

        latest = self.latest_session(namespace=namespace)
        if latest:
            lines.append(f"Last session summary: {latest['summary']}")
            if latest.get("decisions"):
                lines.append("Last session decisions:")
                lines.extend(f"- {item}" for item in latest["decisions"][:limit])

        facts = [
            fact for fact in self.data["facts"]
            if fact.get("namespace", "default") == namespace
        ][-limit:]
        if facts:
            lines.append("Durable facts:")
            lines.extend(f"- {fact['key']}: {fact['value']}" for fact in facts)

        loops = self.active_open_loops(namespace=namespace)[:limit]
        if loops:
            lines.append("Open loops:")
            lines.extend(f"- {loop['text']}" for loop in loops)

        if len(lines) == 2:
            return ""
        return "\n".join(lines)

    def fast_lookup(self, query: str, namespace: str = "default", limit: int = 8) -> str:
        query_l = query.lower()
        hits: List[str] = []

        facts = [
            fact for fact in self.data["facts"]
            if fact.get("namespace", "default") == namespace
        ]
        for fact in reversed(facts):
            haystack = f"{fact['key']} {fact['value']}".lower()
            if any(token in haystack for token in query_l.split()):
                hits.append(f"FACT {fact['key']}: {fact['value']}")
            if len(hits) >= limit:
                break

        for loop in self.active_open_loops(namespace=namespace):
            if any(token in loop["text"].lower() for token in query_l.split()):
                hits.append(f"OPEN LOOP: {loop['text']}")
            if len(hits) >= limit:
                break

        latest = self.latest_session(namespace=namespace)
        if latest:
            if any(phrase in query_l for phrase in ("last session", "resume", "where did we leave off")):
                hits.append(f"LATEST SESSION: {latest['summary']}")
            elif any(token in latest["summary"].lower() for token in query_l.split()):
                hits.append(f"LATEST SESSION: {latest['summary']}")

        return "\n".join(hits[:limit])


class StructuredMemoryBuilder:
    def build_session_record(
        self,
        namespace: str,
        session_id: str,
        label: str,
        turns: List[Dict[str, Any]],
        started_at: Optional[str] = None,
    ) -> SessionSummaryRecord:
        decisions = self.extract_decisions(turns)
        preferences = self.extract_preferences(turns)
        open_loops = self.extract_open_loops(turns)
        summary = self.build_summary(turns, decisions, preferences, open_loops)
        ended_at = datetime.now(timezone.utc).isoformat()
        return SessionSummaryRecord(
            namespace=namespace,
            session_id=session_id,
            label=label,
            started_at=started_at or ended_at,
            ended_at=ended_at,
            summary=summary,
            decisions=decisions,
            preferences=preferences,
            open_loops=open_loops,
        )

    def extract_decisions(self, turns: List[Dict[str, Any]]) -> List[str]:
        items: List[str] = []
        for turn in turns:
            if self._skip_turn(turn):
                continue
            content = turn.get("content", "").strip()
            content_l = content.lower()
            if any(content_l.startswith(prefix) for prefix in DECISION_PREFIXES):
                items.append(content)
        return self._dedupe(items)

    def extract_preferences(self, turns: List[Dict[str, Any]]) -> List[str]:
        items: List[str] = []
        for turn in turns:
            if self._skip_turn(turn):
                continue
            content = turn.get("content", "").strip()
            if any(pattern.search(content) for pattern in PREFERENCE_PATTERNS):
                items.append(content)
        return self._dedupe(items)

    def extract_open_loops(self, turns: List[Dict[str, Any]]) -> List[str]:
        items: List[str] = []
        for turn in turns:
            if self._skip_turn(turn):
                continue
            content = turn.get("content", "").strip()
            if any(pattern.search(content) for pattern in TASK_PATTERNS):
                items.append(content)
        return self._dedupe(items)

    def update_store(
        self,
        store: StructuredMemoryStore,
        session_record: SessionSummaryRecord,
        turns: List[Dict[str, Any]],
    ) -> None:
        store.append_session(session_record)

        for turn in turns:
            if self._skip_turn(turn):
                continue
            content = turn.get("content", "").strip()
            turn_index = turn.get("turn_index", -1)
            facts = self.extract_facts_from_turn(
                content,
                session_record.namespace,
                session_record.session_id,
                turn_index,
            )
            for fact in facts:
                store.upsert_fact(fact)

        for text in session_record.open_loops:
            record = OpenLoopRecord(
                namespace=session_record.namespace,
                id=self._slugify(text),
                text=text,
                status="resolved" if any(pattern.search(text) for pattern in RESOLVED_PATTERNS) else "open",
                source_session_id=session_record.session_id,
                source_turn_index=-1,
            )
            store.upsert_open_loop(record)

    def extract_facts_from_turn(
        self,
        content: str,
        namespace: str,
        session_id: str,
        turn_index: int,
    ) -> List[FactRecord]:
        facts: List[FactRecord] = []
        lowered = content.lower()

        if any(lowered.startswith(prefix) for prefix in DECISION_PREFIXES):
            facts.append(
                FactRecord(
                    namespace=namespace,
                    key="latest_decision",
                    value=content,
                    category="decision",
                    source_session_id=session_id,
                    source_turn_index=turn_index,
                    confidence=0.8,
                )
            )

        pref_match = re.search(r"(?:I prefer|user prefers)\s+(.*)", content, re.IGNORECASE)
        if pref_match:
            facts.append(
                FactRecord(
                    namespace=namespace,
                    key=f"preference:{self._slugify(pref_match.group(1))[:40]}",
                    value=pref_match.group(1).strip(),
                    category="preference",
                    source_session_id=session_id,
                    source_turn_index=turn_index,
                    confidence=0.7,
                )
            )

        final_match = re.search(r"final decision:\s*(.*)", content, re.IGNORECASE)
        if final_match:
            facts.append(
                FactRecord(
                    namespace=namespace,
                    key="final_decision",
                    value=final_match.group(1).strip(),
                    category="decision",
                    source_session_id=session_id,
                    source_turn_index=turn_index,
                    confidence=0.9,
                )
            )

        return facts

    def build_summary(
        self,
        turns: List[Dict[str, Any]],
        decisions: List[str],
        preferences: List[str],
        open_loops: List[str],
    ) -> str:
        pieces: List[str] = []
        if decisions:
            pieces.append(f"Decisions: {'; '.join(decisions[:3])}")
        if preferences:
            pieces.append(f"Preferences: {'; '.join(preferences[:2])}")
        if open_loops:
            pieces.append(f"Open loops: {'; '.join(open_loops[:3])}")
        if not pieces:
            significant = [
                turn.get("content", "").strip()
                for turn in turns
                if turn.get("content", "").strip() and not self._skip_turn(turn)
            ]
            if significant:
                pieces.append(f"Main discussion: {'; '.join(significant[:3])}")
        return " ".join(pieces) if pieces else "No significant updates captured."

    @staticmethod
    def _dedupe(items: List[str]) -> List[str]:
        seen = set()
        result = []
        for item in items:
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(item)
        return result

    @staticmethod
    def _slugify(text: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
        return slug[:80] or "item"

    @staticmethod
    def _skip_turn(turn: Dict[str, Any]) -> bool:
        role = turn.get("role")
        content = turn.get("content", "")
        if role == "system":
            return True
        if content.startswith("[SESSION_BOOTSTRAP]"):
            return True
        return False
