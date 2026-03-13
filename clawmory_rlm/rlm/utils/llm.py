"""
OpenAI Client wrapper specifically for GPT-5 models.
"""

import os
import time
from typing import Optional, Dict, List, Union, Tuple
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Approximate pricing per 1M tokens (update with actual pricing)
PRICING = {
    "gpt-5": {"input": 2.50, "output": 10.00},  # $2.50/$10 per 1M tokens
    "gpt-5-nano": {"input": 0.15, "output": 0.60},  # $0.15/$0.60 per 1M tokens
    "gpt-5-mini": {"input": 0.30, "output": 1.20},  # $0.30/$1.20 per 1M tokens
    "default": {"input": 2.50, "output": 10.00},
}

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
        
        # Track usage
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.call_count = 0
        self.call_history = []  # List of (timestamp, input_tokens, output_tokens, cost, duration)
    
    def get_pricing(self) -> Dict[str, float]:
        """Get pricing for current model."""
        return PRICING.get(self.model, PRICING["default"])
    
    def completion(
        self,
        messages: Union[List[Dict[str, str]], str],
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        try:
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            elif isinstance(messages, dict):
                messages = [messages]

            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=max_tokens,
                **kwargs
            )
            
            duration = time.time() - start_time
            
            # Extract token usage (handle different API versions)
            input_tokens = 0
            output_tokens = 0
            if hasattr(response, 'usage') and response.usage:
                if hasattr(response.usage, 'prompt_tokens'):
                    input_tokens = response.usage.prompt_tokens
                elif hasattr(response.usage, 'input_tokens'):
                    input_tokens = response.usage.input_tokens
                
                if hasattr(response.usage, 'completion_tokens'):
                    output_tokens = response.usage.completion_tokens
                elif hasattr(response.usage, 'output_tokens'):
                    output_tokens = response.usage.output_tokens
                elif hasattr(response.usage, 'total_tokens'):
                    # If only total_tokens available, estimate (rough approximation)
                    total = response.usage.total_tokens
                    # Estimate 80% input, 20% output (rough approximation)
                    input_tokens = int(total * 0.8)
                    output_tokens = total - input_tokens
            
            # Calculate cost
            pricing = self.get_pricing()
            cost = (input_tokens / 1_000_000 * pricing["input"]) + (output_tokens / 1_000_000 * pricing["output"])
            
            # Update totals
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_cost += cost
            self.call_count += 1
            
            # Log this call
            self.call_history.append({
                "timestamp": time.time(),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "cost": cost,
                "duration": duration,
                "model": self.model
            })
            
            return response.choices[0].message.content

        except Exception as e:
            raise RuntimeError(f"Error generating completion: {str(e)}")
    
    def get_usage_summary(self) -> Dict:
        """Get usage summary."""
        return {
            "model": self.model,
            "call_count": self.call_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost": self.total_cost,
            "call_history": self.call_history
        }
    
    def reset_usage(self):
        """Reset usage tracking."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.call_count = 0
        self.call_history = []