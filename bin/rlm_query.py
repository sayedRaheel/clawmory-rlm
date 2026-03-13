import os
import sys
import json

HISTORY_FILE = "/Users/zubair/.openclaw/workspace/clawmory-rlm/history.jsonl"

# Add the parent directory to python path
sys.path.insert(0, "/Users/zubair/.openclaw/workspace/clawmory-rlm")
from clawmory_rlm.chat import MemoryChat

def load_history(chat):
    if not os.path.exists(HISTORY_FILE):
        return
    with open(HISTORY_FILE, 'r') as f:
        for line in f:
            if not line.strip(): continue
            try:
                turn = json.loads(line)
                chat.store.add(turn['role'], turn['content'])
            except:
                pass

def main():
    if len(sys.argv) < 2:
        print("Usage: python rlm_query.py <query>")
        sys.exit(1)
        
    query = " ".join(sys.argv[1:])
    
    # Always force RLM mode for CLI testing. Use local config.
    chat = MemoryChat(
        model=os.getenv("CLAWMORY_MODEL", "gpt-4o-mini"), 
        sub_model=os.getenv("CLAWMORY_MODEL", "gpt-4o-mini"),
        threshold_chars=0
    )
    
    # Load past events
    load_history(chat)
    
    # Run query
    print(f"🧠 Clawmory RLM Engine thinking...")
    answer = chat.query(query)
    
    print("\n===============================")
    print("🎯 EXTRACTION RESULT:")
    print("===============================\n")
    print(answer)

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ OPENAI_API_KEY missing.")
        sys.exit(1)
    main()