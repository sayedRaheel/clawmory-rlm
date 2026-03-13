import os
import time
from typing import Optional, Dict, List, Union
from openai import OpenAI

class OpenAIClient:
    def __init__(self, api_key: Optional[str] = None, model: str = None):
        self.provider = os.getenv("CLAWMORY_PROVIDER", "openai").lower()
        self.model = model or os.getenv("CLAWMORY_MODEL", "gpt-4o-mini")
        
        # For Ollama routing
        if self.provider == "ollama":
            self.base_url = os.getenv("OPENAI_API_BASE", "http://localhost:11434/v1")
            self.api_key = "ollama"
        else:
            self.base_url = None
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
                
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def completion(
        self,
        messages: Union[List[Dict[str, str]], str],
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
            
        # OLLAMA does not consistently support max_completion_tokens
        completion_args = {"model": self.model, "messages": messages}
        if self.provider == "openai" and max_tokens:
            completion_args["max_completion_tokens"] = max_tokens
        elif max_tokens:
            completion_args["max_tokens"] = max_tokens
            
        response = self.client.chat.completions.create(**completion_args, **kwargs)
        
        if hasattr(response, 'usage') and response.usage:
            self.total_input_tokens += getattr(response.usage, 'prompt_tokens', 0)
            self.total_output_tokens += getattr(response.usage, 'completion_tokens', 0)
            
        return response.choices[0].message.content

    def get_usage_summary(self) -> Dict:
        return {
            "model": self.model,
            "provider": self.provider,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
        }
