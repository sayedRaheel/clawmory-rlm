# Clawmory RLM (Recursive Language Model Memory)

This skill provides infinite, scalable conversational memory using the RLM (Recursive Language Model) sub-agent delegation pattern.
Instead of relying on lossy Vector DB RAG, Clawmory spins up a Python REPL and uses parallel sub-agents to precisely extract temporal and factual history from your logs.

## Setup & Use

Clawmory uses a master CLI located at `bin/rlm`.

### 1. Logging a Memory
Whenever a significant decision, preference, or fact occurs in the conversation, log it:
```bash
/Users/zubair/.openclaw/workspace/clawmory-rlm/bin/rlm log assistant "User prefers to build tools in Python, not Go."
/Users/zubair/.openclaw/workspace/clawmory-rlm/bin/rlm log user "My ultimate goal is to build amazing projects on my GitHub that get a ton of stars."
```

### 2. Querying Memory
When the user asks a question about past context, decisions, or timelines, use the RLM engine to extract the exact truth:
```bash
/Users/zubair/.openclaw/workspace/clawmory-rlm/bin/rlm query "What is the user's ultimate goal for their GitHub?"
```

The engine will automatically spin up sub-agents, analyze the history in chunks, and return the precise extracted fact.

## Hybrid Mode (RAG + RLM)
You should use Clawmory `rlm query` *alongside* your standard RAG `memory_search` tool. 
- Use standard `memory_search` for instant keyword reflexes.
- Use `rlm query` for deep logical extraction and to verify/override RAG hallucinations.