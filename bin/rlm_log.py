import sys
import json
import time
from datetime import datetime
import os

HISTORY_FILE = "/Users/zubair/.openclaw/workspace/clawmory-rlm/history.jsonl"

def main():
    if len(sys.argv) < 3:
        print("Usage: python rlm_log.py <role> <content>")
        sys.exit(1)
        
    role = sys.argv[1]
    content = " ".join(sys.argv[2:])
    
    turn = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "role": role,
        "content": content
    }
    
    with open(HISTORY_FILE, "a") as f:
        f.write(json.dumps(turn) + "\n")
        
    print("✅ Logged to Clawmory RLM.")

if __name__ == "__main__":
    main()