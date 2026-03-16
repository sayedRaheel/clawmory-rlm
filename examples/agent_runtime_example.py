from clawmory_rlm.agent_session import AgentSession


def run_example() -> None:
    with AgentSession(
        goal="Resume work on the memory-enabled coding agent",
        history_path="history.jsonl",
        structured_memory_path="structured_memory.json",
        session_label="coding-agent",
    ) as session:
        messages = session.initial_messages()
        print("Initial system prompt:")
        print(messages[0]["content"])

        # Replace this with your actual agent runtime.
        user_message = "What did we decide last time?"
        assistant_message = "We decided to use structured memory with RLM fallback."

        session.log_user(user_message)
        session.log_assistant(assistant_message)

        summary = session.close(label="coding-agent")
        print("\nSession summary:")
        print(summary)


if __name__ == "__main__":
    run_example()
