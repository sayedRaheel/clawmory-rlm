import os
from typing import Optional, Dict, Any

from .rlm.utils.llm import OpenAIClient
from .config import DEFAULT_MODEL, DEFAULT_RLM_TIMEOUT_S
from .memory_router import MemoryRouter
from .memory_store import MemoryStore
from .memory_rlm import MemoryRLM
from .structured_memory import StructuredMemoryStore

class MemoryChat:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        sub_model: str = DEFAULT_MODEL,
        threshold_chars: int = 50000,
        api_key: Optional[str] = None,
        history_path: Optional[str] = None,
        structured_memory_path: Optional[str] = None,
        autoload: bool = True,
        rlm_timeout_s: int = DEFAULT_RLM_TIMEOUT_S,
        namespace: str = "default",
    ):
        self.model = model
        self.sub_model = sub_model
        self.threshold_chars = threshold_chars
        self.rlm_timeout_s = rlm_timeout_s
        self.namespace = namespace
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
        self.store = MemoryStore(history_path=history_path, autoload=autoload)
        self.structured_store = StructuredMemoryStore(path=structured_memory_path)
        
        # Standard chat client
        self.standard_llm = OpenAIClient(api_key=self._api_key, model=self.model)
        
        # The RLM engine for deep memory retrieval
        self._memory_rlm = MemoryRLM(
            model=self.model,
            sub_model=self.sub_model,
            api_key=self._api_key,
            max_workers=5,
            timeout_s=self.rlm_timeout_s,
        )
        self.router = MemoryRouter(self, self.structured_store, namespace=self.namespace)

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
        response = self.router.recall(query)
        return response.content

    def safe_query(self, query: str) -> str:
        try:
            return self.query(query)
        except TimeoutError:
            fast = self.structured_store.fast_lookup(query, namespace=self.namespace)
            if fast:
                return "Deep recall timed out. Structured fallback:\n" + fast
            recent = self.store.current_session_turns()[-6:]
            if recent:
                context = "\n".join(f"[{t['role']}]: {t['content']}" for t in recent)
                return "Deep recall timed out. Recent-history fallback:\n" + context
            return "Deep recall timed out and no fallback memory was available."

    def start_session(self, goal: str, session_label: Optional[str] = None) -> str:
        from .session_memory import SessionMemoryManager

        manager = SessionMemoryManager(chat=self)
        bootstrap = manager.start_session(goal, session_label=session_label)
        return bootstrap.as_system_prompt()
