"""
Simple Recursive Language Model (RLM) with REPL environment.
"""

import time
import json
from typing import Dict, List, Optional, Any 

from rlm import RLM
from rlm.repl import REPLEnv
from rlm.utils.llm import OpenAIClient
from rlm.utils.prompts import DEFAULT_QUERY, next_action_prompt, build_system_prompt
import rlm.utils.utils as utils

from rlm.logger.root_logger import ColorfulLogger
from rlm.logger.repl_logger import REPLEnvLogger


class RLM_REPL(RLM):
    """
    LLM Client that can handle long contexts by recursively calling itself.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 model: str = "gpt-5",
                 recursive_model: str = "gpt-5",
                 max_iterations: int = 20,
                 depth: int = 0,
                 enable_logging: bool = False,
                 ):
        self.api_key = api_key
        self.model = model
        self.recursive_model = recursive_model
        self.llm = OpenAIClient(api_key, model) # Replace with other client
        
        # Track recursive call depth to prevent infinite loops
        self.repl_env = None
        self.depth = depth # Unused in this version.
        self._max_iterations = max_iterations
        
        # Initialize colorful logger
        self.logger = ColorfulLogger(enabled=enable_logging)
        self.repl_env_logger = REPLEnvLogger(enabled=enable_logging)
        
        self.messages = [] # Initialize messages list
        self.query = None
        
        # Track timing and costs
        self.start_time = None
        self.step_timings = []  # List of (iteration, step_name, duration, tokens, cost)
        self.total_duration = 0.0
    
    def setup_context(self, context: List[str] | str | List[Dict[str, str]], query: Optional[str] = None):
        """
        Setup the context for the RLMClient.

        Args:
            context: The large context to analyze in the form of a list of messages, string, or Dict
            query: The user's question
        """
        if query is None:
            query = DEFAULT_QUERY

        self.query = query
        self.logger.log_query_start(query)

        # Initialize the conversation with the REPL prompt
        self.messages = build_system_prompt()
        self.logger.log_initial_messages(self.messages)
        
        # Initialize REPL environment with context data
        context_data, context_str = utils.convert_context_for_repl(context)
        
        self.repl_env = REPLEnv(
            context_json=context_data, 
            context_str=context_str, 
            recursive_model=self.recursive_model,
        )
        
        return self.messages

    def completion(self, context: List[str] | str | List[Dict[str, str]], query: Optional[str] = None) -> str:
        """
        Given a query and a (potentially long) context, recursively call the LM
        to explore the context and provide an answer using a REPL environment.
        """
        self.start_time = time.time()
        self.step_timings = []
        self.llm.reset_usage()
        
        # Reset tracking variables
        self._prev_sub_input_tokens = 0
        self._prev_sub_output_tokens = 0
        self._prev_sub_cost = 0.0
        self._prev_sub_calls = 0
        
        self.messages = self.setup_context(context, query)
        
        # Reset sub-agent usage after repl_env is created
        if self.repl_env and hasattr(self.repl_env, 'sub_rlm') and hasattr(self.repl_env.sub_rlm, 'client'):
            self.repl_env.sub_rlm.client.reset_usage()
        
        # Main loop runs for fixed # of root LM iterations
        for iteration in range(self._max_iterations):
            step_start = time.time()
            
            # Query root LM to interact with REPL environment
            main_llm_start = time.time()
            response = self.llm.completion(self.messages + [next_action_prompt(query, iteration)])
            main_llm_duration = time.time() - main_llm_start
            
            # Get main LLM usage
            main_usage = self.llm.get_usage_summary()
            last_call = main_usage["call_history"][-1] if main_usage["call_history"] else {}
            
            # Check for code blocks
            code_blocks = utils.find_code_blocks(response)
            self.logger.log_model_response(response, has_tool_calls=code_blocks is not None)
            
            # #region agent log
            import json
            try:
                log_path = '/Users/sayedraheel/Recursive_language_model_rlm-minimal/.cursor/debug.log'
                with open(log_path, 'a') as f:
                    f.write(json.dumps({
                        'sessionId': 'debug-session',
                        'runId': 'runtime',
                        'hypothesisId': 'D',
                        'location': 'rlm_repl.py:118',
                        'message': 'Main LLM response analyzed',
                        'data': {
                            'iteration': iteration,
                            'has_code_blocks': code_blocks is not None,
                            'response_contains_llm_query': 'llm_query' in response if response else False,
                            'response_preview': response[:300] if response else None
                        },
                        'timestamp': int(time.time() * 1000)
                    }) + '\n')
            except Exception as e:
                pass
            # #endregion
            
            # Process code execution or add assistant message
            code_exec_start = time.time()
            if code_blocks is not None:
                self.messages = utils.process_code_execution(
                    response, self.messages, self.repl_env, 
                    self.repl_env_logger, self.logger
                )
            else:
                # Add assistant message when there are no code blocks
                assistant_message = {"role": "assistant", "content": "You responded with:\n" + response}
                self.messages.append(assistant_message)
            code_exec_duration = time.time() - code_exec_start
            
            # Get sub-agent usage if available (cumulative up to this point)
            sub_usage = {}
            if self.repl_env and hasattr(self.repl_env, 'sub_rlm') and hasattr(self.repl_env.sub_rlm, 'client'):
                sub_usage = self.repl_env.sub_rlm.client.get_usage_summary()
            
            # Calculate incremental sub-agent usage for this step
            # (sub_usage is cumulative, so we need to track previous totals)
            # Note: _prev_sub_* variables are initialized in completion() method
            
            step_sub_input = sub_usage.get('total_input_tokens', 0) - (getattr(self, '_prev_sub_input_tokens', 0))
            step_sub_output = sub_usage.get('total_output_tokens', 0) - (getattr(self, '_prev_sub_output_tokens', 0))
            step_sub_cost = sub_usage.get('total_cost', 0) - self._prev_sub_cost
            step_sub_calls = sub_usage.get('call_count', 0) - self._prev_sub_calls
            
            # Update previous totals
            self._prev_sub_input_tokens = sub_usage.get('total_input_tokens', 0)
            self._prev_sub_output_tokens = sub_usage.get('total_output_tokens', 0)
            self._prev_sub_cost = sub_usage.get('total_cost', 0)
            self._prev_sub_calls = sub_usage.get('call_count', 0)
            
            # Log step timing
            step_duration = time.time() - step_start
            self.step_timings.append({
                "iteration": iteration + 1,
                "step_name": "main_llm_call",
                "duration": step_duration,
                "main_llm": {
                    "input_tokens": last_call.get("input_tokens", 0),
                    "output_tokens": last_call.get("output_tokens", 0),
                    "cost": last_call.get("cost", 0),
                    "duration": main_llm_duration
                },
                "code_execution": {
                    "duration": code_exec_duration,
                    "has_code": code_blocks is not None
                },
                "sub_agent": {
                    "calls_this_step": step_sub_calls,
                    "input_tokens_this_step": step_sub_input,
                    "output_tokens_this_step": step_sub_output,
                    "cost_this_step": step_sub_cost,
                    "total_calls": sub_usage.get("call_count", 0),
                    "total_input_tokens": sub_usage.get("total_input_tokens", 0),
                    "total_output_tokens": sub_usage.get("total_output_tokens", 0),
                    "total_cost": sub_usage.get("total_cost", 0)
                }
            })
            
            # Check that model produced a final answer
            final_answer = utils.check_for_final_answer(
                response, self.repl_env, self.logger,
            )

            # In practice, you may need some guardrails here.
            if final_answer:
                self.total_duration = time.time() - self.start_time
                self.logger.log_final_response(final_answer)
                self._print_cost_summary()
                return final_answer

            
        # If we reach here, no final answer was found in any iteration
        print("No final answer found in any iteration")
        final_step_start = time.time()
        self.messages.append(next_action_prompt(query, iteration, final_answer=True))
        final_answer = self.llm.completion(self.messages)
        self.total_duration = time.time() - self.start_time
        self.logger.log_final_response(final_answer)
        self._print_cost_summary()

        return final_answer
    
    def _print_cost_summary(self):
        """Print comprehensive cost and timing summary."""
        main_usage = self.llm.get_usage_summary()
        sub_usage = {}
        if self.repl_env and hasattr(self.repl_env, 'sub_rlm') and hasattr(self.repl_env.sub_rlm, 'client'):
            sub_usage = self.repl_env.sub_rlm.client.get_usage_summary()
        
        print("\n" + "="*80)
        print("COST AND TIMING SUMMARY")
        print("="*80)
        
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"  Total Duration: {self.total_duration:.2f}s")
        print(f"  Total Iterations: {len(self.step_timings)}")
        
        print(f"\n🤖 MAIN LLM ({main_usage['model']}):")
        print(f"  Total Calls: {main_usage['call_count']}")
        print(f"  Input Tokens: {main_usage['total_input_tokens']:,}")
        print(f"  Output Tokens: {main_usage['total_output_tokens']:,}")
        print(f"  Total Tokens: {main_usage['total_tokens']:,}")
        print(f"  Cost: ${main_usage['total_cost']:.4f}")
        
        if sub_usage:
            print(f"\n🔧 SUB-AGENT ({sub_usage.get('model', 'unknown')}):")
            print(f"  Total Calls: {sub_usage.get('call_count', 0)}")
            print(f"  Input Tokens: {sub_usage.get('total_input_tokens', 0):,}")
            print(f"  Output Tokens: {sub_usage.get('total_output_tokens', 0):,}")
            print(f"  Total Tokens: {sub_usage.get('total_tokens', 0):,}")
            print(f"  Cost: ${sub_usage.get('total_cost', 0):.4f}")
        
        total_cost = main_usage['total_cost'] + sub_usage.get('total_cost', 0)
        total_tokens = main_usage['total_tokens'] + sub_usage.get('total_tokens', 0)
        
        print(f"\n💰 TOTAL:")
        print(f"  Total Cost: ${total_cost:.4f}")
        print(f"  Total Tokens: {total_tokens:,}")
        print(f"  Cost per Token: ${total_cost/total_tokens*1000:.6f} per 1K tokens" if total_tokens > 0 else "  Cost per Token: N/A")
        
        print(f"\n⏱️  STEP-BY-STEP BREAKDOWN:")
        for step in self.step_timings:
            print(f"\n  Iteration {step['iteration']}:")
            print(f"    Total Step Time: {step['duration']:.2f}s")
            print(f"    Main LLM: {step['main_llm']['input_tokens']:,} in + {step['main_llm']['output_tokens']:,} out = {step['main_llm']['input_tokens'] + step['main_llm']['output_tokens']:,} tokens, ${step['main_llm']['cost']:.4f}, {step['main_llm']['duration']:.2f}s")
            if step['code_execution']['has_code']:
                print(f"    Code Execution: {step['code_execution']['duration']:.2f}s")
            if step['sub_agent'].get('calls_this_step', 0) > 0:
                print(f"    Sub-Agent: {step['sub_agent']['calls_this_step']} calls this step, {step['sub_agent']['input_tokens_this_step']:,} in + {step['sub_agent']['output_tokens_this_step']:,} out = {step['sub_agent']['input_tokens_this_step'] + step['sub_agent']['output_tokens_this_step']:,} tokens, ${step['sub_agent']['cost_this_step']:.4f}")
        
        print("\n" + "="*80)
    
    def cost_summary(self) -> Dict[str, Any]:
        """Get the cost summary of the Root LM + Sub-RLM Calls."""
        main_usage = self.llm.get_usage_summary()
        sub_usage = {}
        if self.repl_env and hasattr(self.repl_env, 'sub_rlm') and hasattr(self.repl_env.sub_rlm, 'client'):
            sub_usage = self.repl_env.sub_rlm.client.get_usage_summary()
        
        return {
            "total_duration": self.total_duration,
            "iterations": len(self.step_timings),
            "main_llm": main_usage,
            "sub_agent": sub_usage,
            "total_cost": main_usage['total_cost'] + sub_usage.get('total_cost', 0),
            "total_tokens": main_usage['total_tokens'] + sub_usage.get('total_tokens', 0),
            "step_timings": self.step_timings
        }

    def reset(self):
        """Reset the (REPL) environment and message history."""
        self.repl_env = REPLEnv()
        self.messages = []
        self.query = None


if __name__ == "__main__":
    pass
