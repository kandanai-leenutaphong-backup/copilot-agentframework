import asyncio

from agent_framework import ChatAgent
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    # Create OpenAI chat client (uses OPENAI_API_KEY environment variable)
    client = OpenAIChatClient(model_id="gpt-4o")

    # Create math expert agent
    math_agent = ChatAgent(
        name="math_expert",
        chat_client=client,
        instructions="You are a math expert.",
        description="A math expert assistant.",
    )

    # Create chemistry expert agent
    chemistry_agent = ChatAgent(
        name="chemistry_expert",
        chat_client=client,
        instructions="You are a chemistry expert.",
        description="A chemistry expert assistant.",
    )

    # Create main assistant agent with expert agents as tools (hierarchical pattern)
    # Agent Framework allows agents to be used directly as tools without wrapper
    agent = ChatAgent(
        name="assistant",
        chat_client=client,
        instructions="You are a general assistant. Use expert tools when needed.",
        tools=[math_agent, chemistry_agent],
    )

    # Execute queries with streaming responses
    print("=== Question 1: Math Query ===")
    async for chunk in agent.run_stream("What is the integral of x^2?"):
        if chunk.text:
            print(chunk.text, end="", flush=True)
    print("\n")

    print("=== Question 2: Chemistry Query ===")
    async for chunk in agent.run_stream("What is the molecular weight of water?"):
        if chunk.text:
            print(chunk.text, end="", flush=True)
    print("\n")


asyncio.run(main())