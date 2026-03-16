---
name: clawmory-rlm
description: |
  Cross-session memory skill for coding agents. Use when an agent needs to resume
  prior work, persist important turns, carry forward open loops or decisions,
  isolate memory by project/thread namespace, or query long history with safe
  fallback behavior. Works through the local Clawmory CLI and Python wrapper.
---

# Clawmory RLM

Use this skill when an agent needs durable memory across sessions.

## When To Use

- Resume an older project/thread with prior decisions and open loops
- Log important turns during a session
- Finalize a session into structured memory
- Query long history safely with timeout + fallback
- Keep memory isolated by project/repo/thread using namespaces

## Core Workflow

Run all commands from the repo root:

```bash
cd /Users/sayedraheel/Desktop/Research_and_Books/testrepo/clawmory-rlm
```

### 1. Start a session

Use `start-session` before the agent begins work. Always pass a namespace for real projects.

```bash
./bin/rlm --namespace clawmory-rlm start-session "Resume work on the memory system"
```

This returns memory to inject as the starting system/context prompt.

### 2. Log important turns

Log decisions, preferences, constraints, and next steps. Do not log every token of casual chatter unless the host runner requires full transcript persistence.

```bash
./bin/rlm --history /tmp/history.jsonl log user "Final decision: use structured memory with RLM fallback."
./bin/rlm --history /tmp/history.jsonl log assistant "Next step: add chronology benchmarks."
```

### 3. Finalize the session

At shutdown, convert the current session into structured memory.

```bash
./bin/rlm --namespace clawmory-rlm end-session --label memory-work
```

This writes:

- session summary
- durable facts
- latest decisions
- open loops

### 4. Deep recall when needed

Use `safe-query` for hard historical questions. It will fall back instead of hanging.

```bash
./bin/rlm --namespace clawmory-rlm safe-query "What was the final backend decision after we changed our mind?"
```

## Recommended Usage Pattern

- Use one namespace per project, repo, or thread
- Use `start-session` once at the beginning of a run
- Log only durable/important turns during the run
- Use `safe-query` for historical questions
- Use `end-session` once before shutdown

## Python Wrapper

If the host agent can call Python directly, prefer `AgentSession`:

```python
from clawmory_rlm import AgentSession

with AgentSession(
    goal="Resume work on the coding agent memory system",
    history_path="history.jsonl",
    structured_memory_path="structured_memory.json",
    session_label="coding-agent",
    namespace="clawmory-rlm",
) as session:
    initial_messages = session.initial_messages()
    session.log_user("Final decision: use structured memory with RLM fallback.")
    session.log_assistant("Next step: add chronology benchmarks.")
```

## Important Notes

- `default` namespace keeps backward-compatible fallback behavior
- named namespaces stay isolated from each other
- deep RLM recall depends on the configured provider/model
- for reliability, prefer OpenAI for deep recall and keep structured memory local

## Files To Know

- `bin/rlm` - main CLI
- `bin/rlm_resume_session.py` - resume helper
- `clawmory_rlm/agent_session.py` - Python runtime wrapper
- `clawmory_rlm/session_memory.py` - session lifecycle
- `clawmory_rlm/structured_memory.py` - durable memory store
