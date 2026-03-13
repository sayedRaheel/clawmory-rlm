"""
prompts.py

System and action prompts for MemoryRLM.
Optimized for local SLMs (e.g. Qwen 3.5 4B).
"""

MEMORY_SYSTEM_PROMPT = """You are a Memory AI. Your job is to extract facts from the conversation history.
You have access to a Python REPL. The chat history is stored in the REPL as `history_turns`.

CRITICAL RULES FOR SMALL MODELS:
1. You MUST write python code in ```python blocks to read the history.
2. Example code to read everything:
```python
print(history_turns)
```
3. Once you read the output and find the answer, you MUST output exactly:
FINAL(the actual answer text here)

Do NOT guess. If you do not know, write python code to search `history_turns` first.
"""

MEMORY_ACTION_PROMPT = (
    'Question: "{query}"\n\n'
    "Step 1: Write python code to search `history_turns` for keywords related to the question.\n"
    "Step 2: Read the output.\n"
    "Step 3: Output FINAL(your answer) based ONLY on the python output.\n"
    "If you cannot find it, output: FINAL(I don't know)\n"
)

MEMORY_FINAL_PROMPT = (
    "Max iterations reached. Provide your best answer to: '{query}'\n"
    "Output exactly: FINAL(your answer here)"
)
