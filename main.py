import asyncio

from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    # Create OpenAI chat client (uses OPENAI_API_KEY environment variable)
    client = OpenAIChatClient(model_id="gpt-4o")

    # Create specialized expert agents
    math_agent = client.create_agent(
        name="math_expert",
        instructions="You are a math expert.",
    )

    chemistry_agent = client.create_agent(
        name="chemistry_expert",
        instructions="You are a chemistry expert.",
    )

    # Convert agents to tools using as_tool() method
    math_tool = math_agent.as_tool(
        name="math_expert_tool",
        description="A math expert assistant.",
        arg_name="query",
        arg_description="The math question to answer",
    )

    chemistry_tool = chemistry_agent.as_tool(
        name="chemistry_expert_tool",
        description="A chemistry expert assistant.",
        arg_name="query",
        arg_description="The chemistry question to answer",
    )

    # Create main agent with expert agents as tools
    agent = client.create_agent(
        name="assistant",
        instructions="You are a general assistant. Use expert tools when needed.",
        tools=[math_tool, chemistry_tool],
    )

    # Execute first query with streaming
    print("=== Query 1: Math Question ===")
    async for chunk in agent.run_stream("What is the integral of x^2?"):
        if chunk.text:
            print(chunk.text, end="", flush=True)
    print("\n")

    # Execute second query with streaming
    print("=== Query 2: Chemistry Question ===")
    async for chunk in agent.run_stream("What is the molecular weight of water?"):
        if chunk.text:
            print(chunk.text, end="", flush=True)
    print("\n")


asyncio.run(main())
