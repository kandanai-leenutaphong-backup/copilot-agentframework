import asyncio
import logging
import os

from azure.ai.projects.aio import AIProjectClient
from azure.identity.aio import DefaultAzureCredential
from azure.ai.projects.models import Agent, AgentThread, MessageRole

async def main():
    # Set up logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    # Initialize the Azure AI Project Client
    # Replace these with your actual Azure AI Project details
    project_connection_string = os.getenv("AZURE_AI_PROJECT_CONNECTION_STRING", "your_connection_string")
    
    async with AIProjectClient.from_connection_string(
        credential=DefaultAzureCredential(),
        conn_str=project_connection_string
    ) as project_client:
        
        # Create an agent with Azure OpenAI
        agent = await project_client.agents.create_agent(
            model="your_models_deployment_name",  # Replace with your model deployment name
            name="ChatAgent",
            instructions="You are a helpful assistant that can control lights.",
            # Note: Tools/functions can be added here if you have a LightsPlugin equivalent
            # tools=[...]
        )
        
        logger.info(f"Created agent with ID: {agent.id}")
        
        # Create a conversation thread
        thread = await project_client.agents.create_thread()
        logger.info(f"Created thread with ID: {thread.id}")
        
        # Initiate a back-and-forth chat
        while True:
            # Collect user input
            user_input = input("User > ")
            
            # Terminate the loop if the user says "exit"
            if user_input == "exit":
                break
            
            # Add user message to the thread
            await project_client.agents.create_message(
                thread_id=thread.id,
                role=MessageRole.USER,
                content=user_input
            )
            
            # Run the agent on the thread
            run = await project_client.agents.create_and_process_run(
                thread_id=thread.id,
                agent_id=agent.id
            )
            
            logger.info(f"Run completed with status: {run.status}")
            
            # Retrieve the agent's response
            messages = await project_client.agents.list_messages(thread_id=thread.id)
            
            # Get the latest assistant message
            for message in messages:
                if message.role == MessageRole.ASSISTANT:
                    # Print the assistant's response
                    for content_item in message.content:
                        if hasattr(content_item, 'text'):
                            print(f"Assistant > {content_item.text.value}")
                    break
        
        # Clean up: delete the agent and thread
        await project_client.agents.delete_agent(agent.id)
        await project_client.agents.delete_thread(thread.id)
        logger.info("Cleaned up agent and thread")

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())