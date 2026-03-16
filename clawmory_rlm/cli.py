from __future__ import annotations

import argparse
import json
from pathlib import Path

from .chat import MemoryChat
from .memory_store import MemoryStore
from .session_memory import SessionMemoryManager
from .structured_memory import StructuredMemoryStore


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clawmory RLM CLI")
    parser.add_argument(
        "--history",
        default=None,
        help="Path to the JSONL history file. Defaults to repo-local history.jsonl.",
    )
    parser.add_argument(
        "--structured-memory",
        default=None,
        help="Path to the structured memory JSON file. Defaults to repo-local structured_memory.json.",
    )
    parser.add_argument(
        "--namespace",
        default="default",
        help="Logical memory namespace, e.g. project/repo/thread name.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    log_parser = subparsers.add_parser("log", help="Append a turn to memory.")
    log_parser.add_argument("role", help="Message role, for example user/assistant/system.")
    log_parser.add_argument("content", nargs="+", help="Turn content to persist.")

    query_parser = subparsers.add_parser("query", help="Query long-term memory.")
    query_parser.add_argument("query", nargs="+", help="Question to answer from memory.")

    safe_query_parser = subparsers.add_parser(
        "safe-query",
        help="Query memory with timeout and graceful fallback.",
    )
    safe_query_parser.add_argument("query", nargs="+", help="Question to answer from memory.")

    start_parser = subparsers.add_parser(
        "start-session",
        help="Retrieve relevant memory at the beginning of a new session.",
    )
    start_parser.add_argument("goal", nargs="+", help="What this new session needs to continue.")
    start_parser.add_argument("--label", default=None, help="Optional session label.")

    end_parser = subparsers.add_parser(
        "end-session",
        help="Summarize the current session into durable structured memory.",
    )
    end_parser.add_argument("--label", default=None, help="Optional session label override.")

    dump_parser = subparsers.add_parser("dump", help="Print raw history as JSON.")
    dump_parser.add_argument("--limit", type=int, default=0, help="Optional tail limit.")

    dump_structured_parser = subparsers.add_parser("dump-structured", help="Print structured memory store.")

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "log":
        store = MemoryStore(history_path=args.history, autoload=True)
        store.add(args.role, " ".join(args.content))
        print(f"Logged to {store.history_path}")
        return 0

    if args.command == "query":
        chat = MemoryChat(
            history_path=args.history,
            structured_memory_path=args.structured_memory,
            namespace=args.namespace,
        )
        print(chat.query(" ".join(args.query)))
        return 0

    if args.command == "safe-query":
        chat = MemoryChat(
            history_path=args.history,
            structured_memory_path=args.structured_memory,
            namespace=args.namespace,
        )
        print(chat.safe_query(" ".join(args.query)))
        return 0

    if args.command == "start-session":
        manager = SessionMemoryManager(
            history_path=args.history,
            structured_memory_path=args.structured_memory,
            namespace=args.namespace,
        )
        bootstrap = manager.start_session(" ".join(args.goal), session_label=args.label)
        print(bootstrap.as_system_prompt())
        return 0

    if args.command == "end-session":
        manager = SessionMemoryManager(
            history_path=args.history,
            structured_memory_path=args.structured_memory,
            namespace=args.namespace,
        )
        print(manager.end_session(label=args.label))
        return 0

    if args.command == "dump":
        store = MemoryStore(history_path=args.history, autoload=True)
        rows = store.to_dicts()
        if args.limit > 0:
            rows = rows[-args.limit:]
        print(json.dumps(rows, indent=2))
        return 0

    if args.command == "dump-structured":
        store = StructuredMemoryStore(path=args.structured_memory)
        print(json.dumps(store.data, indent=2))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
