
from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent, RoutingAgent
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

persona = "You are a college professor"

knowledge_texas = "You know everything about Texas"
texas_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona,
    knowledge=knowledge_texas)

knowledge_europe = "You know everything about Europe"
europe_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona,
    knowledge=knowledge_europe)
# TODO: 3 - Define the Europe Knowledge Augmented Prompt Agent

persona_math = "You are a college math professor"
knowledge_math = "You know everything about math, you take prompts with numbers, extract math formulas, and show the answer without explanation"
math_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_math,
    knowledge=knowledge_math)

agents = [
    {
        "name": "texas agent",
        "description": "Answers a question about Texas",
        "func": texas_agent.respond
    },
    {
        "name": "europe agent",
        "description": "Answers question about Europe",
        "func": europe_agent.respond 
    },
    {
        "name": "math agent",
        "description": "When a prompt contains numbers, respond with a math formula extracted from the prompt",
        "func": math_agent.respond
    }
]
routing_agent = RoutingAgent(openai_api_key, agents=agents)

# TODO: 8 - Print the RoutingAgent responses to the following prompts:
#           - "Tell me about the history of Rome, Texas"
#           - "Tell me about the history of Rome, Italy"
#           - "One story takes 2 days, and there are 20 stories"
prompt_texas = "Tell me about the history of Rome, Texas"
prompt_europe = "Tell me about the history of Rome, Italy"
prompt_math = "One story takes 2 days, and there are 20 stories"

print(f"RoutingAgent response to '{prompt_texas}': {routing_agent.route(prompt_texas)}")
print("-" * 50)
print(f"RoutingAgent response to '{prompt_europe}': {routing_agent.route(prompt_europe)}")
print("-" * 50)
print(f"RoutingAgent response to '{prompt_math}': {routing_agent.route(prompt_math)}")