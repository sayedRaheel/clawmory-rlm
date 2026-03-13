import sys
import json
from clawmory_rlm.chat import MemoryChat
from clawmory_rlm.memory_store import MemoryStore

def test_rlm(query):
    print("🧠 Clawmory RLM Engine thinking...")
    
    # Force use a smaller chunking threshold so it actually triggers RLM instead of direct reading
    chat = MemoryChat(threshold_chars=500)
    
    # Load history
    with open('history.jsonl', 'r') as f:
        for line in f:
            if line.strip():
                try:
                    msg = json.loads(line)
                    chat.store.add(msg.get("role", "user"), msg.get("content", ""))
                except:
                    pass
                    
    ans = chat.query(query)
    print("=" * 31)
    print("🎯 EXTRACTION RESULT:")
    print("=" * 31)
    print(ans)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_rlm(sys.argv[1])
    else:
        print("Provide a query.")
