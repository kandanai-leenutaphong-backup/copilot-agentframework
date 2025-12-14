import asyncio

from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    # Create OpenAI chat client (uses OPENAI_API_KEY environment variable)
    client = OpenAIChatClient(model_id="gpt-4o")

    # Create math expert agent
    math_agent = client.create_agent(
        name="math_expert",
        instructions="You are a math expert.",
    )

    # Create chemistry expert agent
    chemistry_agent = client.create_agent(
        name="chemistry_expert",
        instructions="You are a chemistry expert.",
    )

    # Convert agents to tools using as_tool() method (hierarchical pattern)
    math_tool = math_agent.as_tool(
        name="math_expert",
        description="A math expert assistant.",
        arg_name="question",
        arg_description="The math question to ask",
    )

    chemistry_tool = chemistry_agent.as_tool(
        name="chemistry_expert",
        description="A chemistry expert assistant.",
        arg_name="question",
        arg_description="The chemistry question to ask",
    )

    # Create main assistant agent with expert agents as tools (hierarchical pattern)
    agent = client.create_agent(
        name="assistant",
        instructions="You are a general assistant. Use expert tools when needed.",
        tools=[math_tool, chemistry_tool],
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
