# agentic_workflow.py
from workflow_agents.base_agents import (
    ActionPlanningAgent,
    KnowledgeAugmentedPromptAgent,
    EvaluationAgent,
    RoutingAgent,
)

import os
from dotenv import load_dotenv

load_dotenv()
open_ai_key = os.getenv("OPENAI_API_KEY")

########################
# load the product spec
########################
file_name = "Product-Spec-Email-Router.txt"
try: 
    with open(file_name, "r", encoding='utf-8') as file:
        product_spec = file.read()

except FileNotFoundError:
    print(f"Error: The file '{file_name}' was not found.")
except Exception as e:
    print(f"An error occurred while reading the file: {e}")


########################
# Instantiate all the agents
########################

########################
# Action Planning Agent
########################
knowledge_action_planning = """
    # Development Plan for Product Management Workflow
    1. Define user stories based on the product specification.
    2. Group user stories into features.
    3. Define development tasks for each feature.
"""
action_planning_agent = ActionPlanningAgent(
    openai_api_key=open_ai_key,
    knowledge=knowledge_action_planning
)

########################
# Product Manager - Knowledge Augmented Prompt Agent
########################
persona_product_manager = "You are a Product Manager, you are responsible for defining the user stories for a product."
knowledge_product_manager = f"""
    User stories are defined by writing sentences with a persona, an action, and a desired outcome.
    The sentences always start with: As a 
    Write several stories for the product spec below, where the personas are the different users of the product. 
    New user stories should start on a new line. 
    The product spec is:\n {product_spec} \n
    """
product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=open_ai_key,
    persona=persona_product_manager,
    knowledge=knowledge_product_manager
)

########################
# Product Manager - Evaluation Agent
########################
persona_product_manager_eval = "You are an evaluation agent that checks the answers of other worker agents."
evaluation_criteria_product_manager = "All user stories should follow the structure: 'As a [type of user], I want [an action or feature] so that [benefit/value].' "
max_interactions = 10
product_manager_evaluation_agent = EvaluationAgent(
    openai_api_key=open_ai_key,
    persona=persona_product_manager_eval,
    evaluation_criteria=evaluation_criteria_product_manager,
    worker_agent=product_manager_knowledge_agent,
    max_interactions=max_interactions
)

########################
# Program Manager - Knowledge Augmented Prompt Agent
########################
persona_program_manager = "You are a Program Manager, you are responsible for defining the features for a product."
knowledge_program_manager = "Features of a product are defined by organizing similar user stories into cohesive groups."
program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=open_ai_key,
    persona=persona_program_manager,
    knowledge=knowledge_program_manager
)

########################
# Program Manager - Evaluation Agent
########################
persona_program_manager_eval = "You are an evaluation agent that checks the answers of other worker agents."
evaluation_criteria_program_manager = """
The answer should be product features that follow the following structure:
Feature Name: A clear, concise title that identifies the capability\n
Description: A brief explanation of what the feature does and its purpose\n
Key Functionality: The specific capabilities or actions the feature provides\n
User Benefit: How this feature creates value for the user
"""
program_manager_evaluation_agent = EvaluationAgent(
    openai_api_key=open_ai_key,
    persona=persona_program_manager_eval,
    evaluation_criteria=evaluation_criteria_program_manager,
    worker_agent=program_manager_knowledge_agent,
    max_interactions=max_interactions
)


########################
# Development Engineer - Knowledge Augmented Prompt Agent
########################
persona_dev_engineer = "You are a Development Engineer, you are responsible for defining the development tasks for a product."
knowledge_dev_engineer = "Development tasks are defined by identifying what needs to be built to implement each user story, including acceptance criteria, effort estimates, and dependencies. Make sure to include tasks for all features in your response, not just for a subset of the features."
development_engineer_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=open_ai_key,
    persona=persona_dev_engineer,
    knowledge=knowledge_dev_engineer
)

########################
# Development Engineer - Evaluation Agent
########################
persona_dev_engineer_eval = "You are an evaluation agent that checks the answers of other worker agents."
evaluation_criteria_dev_engineer = """
The answer should be tasks following this exact structure: 
- Task ID: A unique identifier for tracking purposes\n
- Task Title: Brief description of the specific development work\n
- Related User Story: Reference to the parent user story\n
- Description: Detailed explanation of the technical work required\n
- Acceptance Criteria: Specific requirements that must be met for completion\n
- Estimated Effort: Time or complexity estimation\n
- Dependencies: Any tasks that must be completed first

The answer should contain tasks related to all the features as passed in the initial prompt, not just a subset of the features.
"""
development_engineer_evaluation_agent = EvaluationAgent(
    openai_api_key=open_ai_key,
    persona=persona_dev_engineer_eval,
    evaluation_criteria=evaluation_criteria_dev_engineer,
    worker_agent=development_engineer_knowledge_agent,
    max_interactions=max_interactions
)

########################
# Routing Agent
########################
# TODO: 11 - Define the support functions for the routes of the routing agent (e.g., product_manager_support_function, program_manager_support_function, development_engineer_support_function).
# Each support function should:
#   1. Take the input query (e.g., a step from the action plan).
#   2. Get a response from the respective Knowledge Augmented Prompt Agent.
#   3. Have the response evaluated by the corresponding Evaluation Agent.
#   4. Return the final validated response.
# Comment: 
# - the ***_evaluation_agent.evaluate() method will be used to evaluate the response of the ***_knowledge_agent.respond() method
# - besides, the ***_evaluation_agent.evaluate() methods expect an initial user prompt as input, not the response of the ***_knowledge_agent.respond() method 
#   - this was defined in phase 1, and is also demanded by the project rubric. 
#   - if one really wants to call the '.respond()' functions, first, one could change the '.evaluate()' method to accept an optional 'isworkerresponse' parameter to indicate that the `initial_prompt` is actually a response from the worker agent, and not a user prompt.
# - therefore, the support functions do not invoke the ***_knowledge_agent.respond() method, first, but rather call the ***_evaluation_agent.evaluate() method

# Job function persona support functions
def product_manager_support_function(query):
    """
    Support function for the Product Manager agent.
    Takes a query, then gets a response from the Product Manager Knowledge Agent (via the evaluation agent),
    evaluates it with the Product Manager Evaluation Agent, and returns the final response.
    """
    evaluation_result = product_manager_evaluation_agent.evaluate(query)
    return evaluation_result['final_response']

def program_manager_support_function(query):
    """
    Support function for the Program Manager agent.
    Takes a query, then gets a response from the Program Manager Knowledge Agent (via the evaluation agent),
    evaluates it with the Program Manager Evaluation Agent, and returns the final response.
    """
    evaluation_result = program_manager_evaluation_agent.evaluate(query)
    return evaluation_result['final_response']

def development_engineer_support_function(query):
    """
    Support function for the Development Engineer agent.
    Takes a query, then gets a response from the Development Engineer Knowledge Agent (via the evaluation agent),
    evaluates it with the Development Engineer Evaluation Agent, and returns the final response.
    """
    evaluation_result = development_engineer_evaluation_agent.evaluate(query)
    return evaluation_result['final_response']

agents = [
    {
        "name": "product manager",
        "description": "The product manager agent defines user stories based on the product specification",
        "func": product_manager_support_function
    },
    {
        "name": "program manager",
        "description": "The program manager agent defines features for the product based on user stories",
        "func": program_manager_support_function
    },
    {
        "name": "development engineer",
        "description": "The development engineer agent defines development tasks for each feature of the product",
        "func": development_engineer_support_function
    }
]
routing_agent = RoutingAgent(open_ai_key, agents=agents)

########################
# Run the workflow
########################

print("\n*** Workflow execution started ***\n")
# Workflow Prompt
workflow_prompt = "What would the development tasks for this product be?"
print(f"Task to complete in this workflow, workflow prompt = {workflow_prompt}")
print("-" * 50)

print("\nDefining workflow steps from the workflow prompt")
response = action_planning_agent.extract_steps_from_prompt(workflow_prompt)
assert isinstance(response, list), "The response should be a list of steps."
# Concatenate the steps into a single string for better readability
response_string = "\n".join(response) 
print(f"ActionPlanningAgent response to workflow prompt:\n{response_string}")
print("-" * 50)

completed_steps = []
for i in range(len(response)):
    step = response[i]
    print("-" * 50)
    print(f"\nExecuting step: {step}")
    # Route the step and any prior output to the appropriate support function using the routing agent
    if i > 0:
        # If this is not the first step, include the last completed step as context
        routing_query = f"""Input from previous step:
        {completed_steps[-1]}
        The step you should perform: 
        {step}
"""
    else:
        # If this is the first step, just use the step itself
        routing_query = step
    result = routing_agent.route(routing_query)
    print("-" * 20)
    completed_steps.append(result)
    print(f"Result of step '{step}': {result}")
    print("-" * 50)

# Print the final output of the workflow
print("\n*** Workflow execution completed ***\n")
print("Final output of the workflow:")
if completed_steps:
    print(completed_steps[-1])  # Print the last completed step
else:
    print("No steps were completed in the workflow.")


########################
### REAMAINING TODOS ### 
########################

# Phase 2:
# Potentially change the final output of this python file to summarize all user stories, features, and development tasks.

# Generate summary
print("-" * 50)
print("=== Summary of Completed Steps ===")
for i in range(min(len(completed_steps), len(response))):
    print("-" * 20)
    print(f"Step {i+1}: {response[i]}\n{completed_steps[i]}")