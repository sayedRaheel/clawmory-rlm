![Clawmory RLM](./docs/clawmory-banner.png)

# Clawmory RLM 🧠

**Scalable conversational memory via Recursive Sub-Agent Delegation.**

Clawmory RLM is a lightweight, local-first Python engine for durable agent memory and long-history retrieval. Instead of relying only on chunk similarity, Clawmory uses a **Recursive Language Model pattern**: it spins up a Python REPL and uses parallel sub-agents to extract temporal and factual history from chat logs.

It is optimized for small local models and supports Ollama, so your data can stay on your machine and API costs can stay at $0.

## Why RAG is Dead for Agent Timelines ❌
If you use standard semantic search (Vector Databases) on a 10,000-line chat history, the AI blindly grabs 5 chunks that "sound similar" to your query. If the user changed their mind mid-conversation, the RAG chunks will contradict each other, and the agent hallucinates.

## How Clawmory RLM Fixes It ✅
When a user asks a complex question about the past:
1. Clawmory breaks your massive JSONL history into sequential chunks.
2. It spins up parallel, lightweight SLM sub-agents to read each chunk sequentially via Python.
3. They mathematically extract the timeline and deduce the *chronological truth*.
4. You get perfect recall with 0% data leakage.

## Installation

```bash
git clone https://github.com/sayedRaheel/clawmory-rlm.git
cd clawmory-rlm
python3 -m pip install -e .
```

## Configuration
Create a `.env` file in the root directory. We strongly recommend local Ollama for zero-cost, private inference.

```env
# Example: Using local Ollama
CLAWMORY_PROVIDER=ollama
OPENAI_API_BASE=http://localhost:11434/v1
OPENAI_API_KEY=ollama
CLAWMORY_MODEL=gpt-oss:20b
CLAWMORY_RLM_TIMEOUT_S=20
```

Use a model name that actually exists in `ollama list`.

You can also use OpenAI directly:

```env
CLAWMORY_PROVIDER=openai
OPENAI_API_KEY=your_real_openai_key
CLAWMORY_MODEL=gpt-4o-mini
CLAWMORY_RLM_TIMEOUT_S=20
```

## CLI

The repo now includes a portable `rlm` CLI with no hardcoded machine paths.

```bash
./bin/rlm log user "We decided to keep the backend in Python."
./bin/rlm query "What did we decide about the backend?"
./bin/rlm safe-query "What was the final backend decision after we changed our mind?"
./bin/rlm start-session "Resume the backend architecture discussion"
./bin/rlm end-session --label "backend-architecture"
./bin/rlm dump --limit 5
./bin/rlm dump-structured
python3 bin/rlm_resume_session.py "Resume the backend architecture discussion"
```

`start-session` is the key cross-session workflow. It:

1. creates a new session marker in the JSONL history
2. loads structured memory first: latest summary, durable facts, and open loops
3. falls back to deep RLM recall when structured memory is insufficient
4. returns text you can inject as a system prompt at the beginning of a new agent run

`end-session` turns the current session into durable structured memory:

- session summary
- decisions
- durable preferences/facts
- open loops / next steps

This makes the next startup cheap and reliable instead of forcing a full-history deep query every time.

`safe-query` is the hardened deep-recall path. It times out instead of hanging forever and falls back to structured memory or recent history when the deep model path is unavailable.

## Usage (Python API)
Drop Clawmory into your own AI applications with just a few lines:

```python
from clawmory_rlm.chat import MemoryChat

# Initialize the engine and auto-load persisted history
chat = MemoryChat(threshold_chars=500, history_path="history.jsonl")

# Log history
chat.store.add("user", "We decided to build the backend in Python.")
chat.store.add("assistant", "Noted. Python backend.")
chat.store.add("user", "Actually, let's use Go instead. Wait, no, TypeScript.")
chat.store.add("assistant", "Noted.")
chat.store.add("user", "Final decision: We are doing Python with FastAPI.")

# Query the engine
answer = chat.query("What language did we finally decide to use for the backend?")
print(answer) # "Python with FastAPI"
```

## Cross-Session Memory

If you want a future session to resume with memory from earlier sessions, wire in `start_session()` before normal interaction:

```python
from clawmory_rlm import MemoryChat

chat = MemoryChat(history_path="history.jsonl")
bootstrap = chat.start_session("Continue work on the coding agent memory system")

messages = [
    {"role": "system", "content": bootstrap},
    {"role": "user", "content": "What were we doing last time?"},
]
```

Clawmory does not automatically modify your external agent runtime. Your runner must call this at session start and log turns as they happen.

## Agent Runtime Wrapper

If you want a thin integration layer instead of calling the CLI manually, use `AgentSession`:

```python
from clawmory_rlm import AgentSession

with AgentSession(
    goal="Resume work on the coding agent memory system",
    history_path="history.jsonl",
    structured_memory_path="structured_memory.json",
    session_label="coding-agent",
) as session:
    initial_messages = session.initial_messages()
    # Inject initial_messages into your agent runtime.

    session.log_user("What did we decide last time?")
    session.log_assistant("We decided to use structured memory with RLM fallback.")
```

There is also a runnable example in [examples/agent_runtime_example.py](/Users/sayedraheel/Desktop/Research_and_Books/testrepo/clawmory-rlm/examples/agent_runtime_example.py).

## Quickstart For Agent Hosts

### OpenClaw

OpenClaw-style local runners are the easiest fit. Use the skill or local Python wrapper:

1. Run `start-session` before the agent begins
2. Inject the returned memory as the initial system/context prompt
3. Log durable turns during the run
4. Run `end-session` on shutdown

### Codex / Claude Code

If the host can run local shell commands, use the CLI:

```bash
./bin/rlm --namespace my-project start-session "Resume work on my-project"
./bin/rlm --namespace my-project log user "Final decision: use FastAPI."
./bin/rlm --namespace my-project safe-query "What did we finally decide?"
./bin/rlm --namespace my-project end-session --label work
```

If the host can run Python directly, prefer `AgentSession`.

### Custom Python Agents

This is the cleanest integration path:

```python
from clawmory_rlm import AgentSession

with AgentSession(goal="Resume work", namespace="my-project") as session:
    system_prompt = session.state.system_prompt
    # inject system_prompt into the agent
```

## Infinite Context, Practically

Clawmory is useful as a practical infinite-context layer for agent history:

- append every important turn to `history.jsonl`
- maintain structured memory in `structured_memory.json` at session boundaries
- call `query()` or `start_session()` when the live context window is too small
- inject the retrieved result back into the active prompt

That gives you retrieval over arbitrarily long history without stuffing the full transcript into the model context.

## Recommended Agent Loop

1. On every turn, append raw events to `history.jsonl`.
2. At session end, call `end_session()` to extract durable memory.
3. At the next session start, call `start_session(goal)` to preload:
   - latest session summary
   - open loops
   - durable facts and preferences
4. Use deep RLM recall only for hard chronology or contradiction questions.

## Architecture
Based on the [RLM-Memory paper](https://github.com/sayedRaheel/rlm-memory), this engine dynamically evaluates the complexity of the query. If the history is massive, it isolates chunks of the timeline to deduce the logical truth, achieving much higher accuracy (46% EM) than traditional Vector RAG (5% EM) on long-context tasks.

---
*Created by [Sayed Raheel] for the OpenClaw / AI Agent community.*
