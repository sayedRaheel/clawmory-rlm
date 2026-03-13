import os
import sys
from typing import Optional, Dict, Any

from .rlm.utils.llm import OpenAIClient
from .memory_store import MemoryStore
from .memory_rlm import MemoryRLM

class MemoryChat:
    def __init__(
        self,
        model: str = os.getenv("CLAWMORY_MODEL", "qwen3.5:4b"),
        sub_model: str = os.getenv("CLAWMORY_MODEL", "qwen3.5:4b"),
        threshold_chars: int = 50000,
        api_key: Optional[str] = None
    ):
        self.model = model
        self.sub_model = sub_model
        self.threshold_chars = threshold_chars
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
        self.store = MemoryStore()
        
        # Standard chat client
        self.standard_llm = OpenAIClient(api_key=self._api_key, model=self.model)
        
        # The RLM engine for deep memory retrieval
        self._memory_rlm = MemoryRLM(
            model=self.model,
            sub_model=self.sub_model,
            api_key=self._api_key,
            max_workers=5
        )

    def add_turn(self, role: str, content: str):
        self.store.add(role, content)

    def send(self, message: str) -> str:
        self.store.add("user", message)
        history_chars = self.store.total_chars()
        use_rlm = history_chars >= self.threshold_chars

        if use_rlm:
            response = self._rlm_response(message)
        else:
            response = self._standard_response()
            
        self.store.add("assistant", response)
        return response

    def _standard_response(self) -> str:
        messages = [
            {"role": t["role"], "content": t["content"]}
            for t in self.store.turns
        ]
        return self.standard_llm.completion(messages)

    def _rlm_response(self, query: str) -> str:
        return self._memory_rlm.completion(
            history=self.store,
            query=query,
        )

    def query(self, query: str) -> str:
        return self._rlm_response(query)
