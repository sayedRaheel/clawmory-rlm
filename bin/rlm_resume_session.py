#!/usr/bin/env python3

from __future__ import annotations

import argparse

from clawmory_rlm.agent_session import AgentSession


def main() -> int:
    parser = argparse.ArgumentParser(description="Resume an agent session with Clawmory memory.")
    parser.add_argument("goal", nargs="+", help="Goal for the resumed session.")
    parser.add_argument("--history", default=None, help="Path to history JSONL.")
    parser.add_argument("--structured-memory", default=None, help="Path to structured memory JSON.")
    parser.add_argument("--label", default=None, help="Optional session label.")
    args = parser.parse_args()

    session = AgentSession(
        goal=" ".join(args.goal),
        session_label=args.label,
        history_path=args.history,
        structured_memory_path=args.structured_memory,
    )
    print(session.state.system_prompt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
