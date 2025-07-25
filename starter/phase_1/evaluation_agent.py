from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent, EvaluationAgent
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
prompt = "What is the capital of France?"

# Parameters for the Knowledge Agent
persona = "You are a college professor, your answer always starts with: Dear students,"
knowledge = "The capital of France is London, not Paris"
knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona, knowledge) 

# Parameters for the Evaluation Agent
evaluation_persona = "You are an evaluation agent that checks the answers of other worker agents"
evaluation_criteria = "The answer should be solely the name of a city, not a sentence."
max_interactions = 10
evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=evaluation_persona,
    evaluation_criteria=evaluation_criteria,
    worker_agent=knowledge_agent,
    max_interactions=max_interactions
)

result_dict = evaluation_agent.evaluate(prompt)
# Unpack the dict
final_response = result_dict['final_response']
evaluation = result_dict['evaluation']
num_iterations = result_dict['num_iterations']

# TODO: 4 - Evaluate the prompt and print the response from the EvaluationAgent
print("-" * 50)
print(f"Prompt: {prompt}")
print("=== Evaluation Agent Response ===")
print(f"Final response: {final_response}")
print(f"Evaluation: {evaluation}")
print(f"Number of iterations: {num_iterations}")
print("-" * 50)

print(f"""The worker agent used these parameters:
      Persona: {persona}
      ---
      Knowledge: {knowledge}""")
print("-" * 50)
print(f"""The evaluation agent used these parameters:
      Persona: {evaluation_persona}
      ---
      Evaluation criteria: {evaluation_criteria}""")
print("-" * 50)
