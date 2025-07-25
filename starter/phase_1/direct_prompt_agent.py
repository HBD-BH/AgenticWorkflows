# Test script for DirectPromptAgent class

from workflow_agents.base_agents import DirectPromptAgent
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the Capital of France?"

direct_agent = DirectPromptAgent(
    openai_api_key,
    # base_url="https://openai.vocareum.com/v1",
    # model="gpt-4.1-nano"
)
direct_agent_response = direct_agent.respond(prompt)

# Print the response from the agent
print("-" * 50)
print(f"Prompt: {prompt}")
print("=== Direct Prompt Agent Response ===")
print(direct_agent_response)
print("-" * 50)

# TODO: 5 - Print an explanatory message describing the knowledge source used by the agent to generate the response
print("The response was generated using the OpenAI API with the provided prompt. No system prompt or other context was used.")
print("-" * 50)
