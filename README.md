![Clawmory RLM](./docs/clawmory-banner.png)

# Clawmory RLM 🧠

**Scalable Conversational Memory via Recursive Sub-Agent Delegation.**

Clawmory RLM is a lightweight, local-first Python engine that gives your AI agents infinite memory. Instead of relying on lossy Vector DBs (RAG) that suffer from concept drift, Clawmory uses a **Recursive Language Model pattern**: it spins up a Python REPL and uses parallel sub-agents to precisely extract temporal and factual history from your chat logs.

It is heavily optimized for Small Language Models (SLMs) and supports local, free execution via Ollama (e.g., `qwen3.5:4b`), meaning your data never leaves your machine and your API costs are $0.

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
pip install -e .
```

## Configuration
Create a `.env` file in the root directory. We strongly recommend local Ollama for zero-cost, private inference.

```env
# Example: Using Local Ollama (Free & Private)
CLAWMORY_PROVIDER=ollama
OPENAI_API_BASE=http://localhost:11434/v1
OPENAI_API_KEY=ollama
CLAWMORY_MODEL=qwen3.5:4b
```

## Usage (Python API)
Drop Clawmory into your own AI applications with just a few lines:

```python
from clawmory_rlm.chat import MemoryChat

# Initialize the engine (it reads your .env configuration automatically)
chat = MemoryChat(threshold_chars=500) # Threshold dictates when to trigger recursive sub-agents

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

## Architecture
Based on the [RLM-Memory paper](https://github.com/sayedRaheel/rlm-memory), this engine dynamically evaluates the complexity of the query. If the history is massive, it isolates chunks of the timeline to deduce the logical truth, achieving much higher accuracy (46% EM) than traditional Vector RAG (5% EM) on long-context tasks.

---
*Created by [Sayed Raheel] for the OpenClaw / AI Agent community.*
