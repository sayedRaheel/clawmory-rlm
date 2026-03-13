# Clawmory RLM

**Scalable Conversational Memory via Recursive Sub-Agent Delegation.**

Clawmory RLM is a lightweight, local-first Python engine that gives your AI agents infinite memory. Instead of relying on lossy Vector DBs (RAG) that suffer from concept drift, Clawmory uses a Recursive Language Model pattern: it spins up a Python REPL and uses parallel sub-agents to precisely extract temporal and factual history from your chat logs.

It supports local, free execution via Ollama (e.g., Qwen 3.5), meaning your data never leaves your machine and your API costs are $0.

## Installation

```bash
git clone https://github.com/sayedRaheel/clawmory-rlm.git
cd clawmory-rlm
pip install -e .
```

## Configuration

Create a `.env` file in the root directory:

```env
# Example: Using Local Ollama (Free & Private)
CLAWMORY_PROVIDER=ollama
OPENAI_API_BASE=http://localhost:11434/v1
OPENAI_API_KEY=ollama
CLAWMORY_MODEL=qwen3.5:4b

# Example: Using OpenAI
# CLAWMORY_PROVIDER=openai
# OPENAI_API_KEY=sk-proj-...
# CLAWMORY_MODEL=gpt-4o-mini
```

## Usage (CLI)

Clawmory comes with a built-in CLI to easily log facts and query your history.

**1. Log a memory:**
```bash
./bin/rlm log user "My ultimate goal is to build amazing projects that get GitHub stars."
```

**2. Query your memory:**
```bash
./bin/rlm query "What is the user's ultimate goal?"
```

## Usage (Python API)

You can easily drop Clawmory into your own AI applications:

```python
from clawmory_rlm.chat import MemoryChat

# Initialize the engine (it reads your .env configuration automatically)
chat = MemoryChat(threshold_chars=0)

# Log history
chat.store.add("user", "We decided to build the backend in Python.")
chat.store.add("assistant", "Noted. Python backend.")
chat.store.add("user", "Actually, let's use Go instead.")

# Query the engine
answer = chat.query("What language did we finally decide to use for the backend?")
print(answer) # "Go"
```

## How it works
Based on the [RLM-Memory paper](https://github.com/sayedRaheel/rlm-memory), this engine dynamically evaluates the complexity of the query. If the history is small, it reads it directly. If the history is massive, it uses parallel sub-agents to scan isolated chunks of the timeline to deduce the logical truth, achieving much higher accuracy (46% EM) than traditional Vector RAG (5% EM) on long-context tasks.

---
*Created for the OpenClaw / AI Agent community.*
