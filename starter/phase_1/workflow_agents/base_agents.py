from openai import OpenAI
import numpy as np
import pandas as pd
import re
import csv
import uuid
from datetime import datetime


# DirectPromptAgent class definition
class DirectPromptAgent:
    """
    An agent that directly responds to user prompts without any additional context or system prompts.
    """

    def __init__(self, openai_api_key, base_url="https://openai.vocareum.com/v1", model="gpt-4.1-nano"):
        """
        Initializes the DirectPromptAgent with API key, base URL, and model.

        Args:
            openai_api_key (str): The API key for OpenAI.
            base_url (str, optional): The base URL for the OpenAI API.
                                      Defaults to "https://openai.vocareum.com/v1".
            model (str, optional): The model to use for the API calls.
                                   Defaults to "gpt-4.1-nano".
        """
        self.openai_api_key = openai_api_key
        self.base_url = base_url
        self.model = model

    def respond(self, prompt):
        # Generate a response using the OpenAI API
        client = OpenAI(
            api_key=self.openai_api_key,
            base_url=self.base_url
            )
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                # No system prompt for the DirectPromptAgent
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip()

        
# AugmentedPromptAgent class definition
class AugmentedPromptAgent:
    """
    An agent that uses a persona to generate responses based on user prompts.
    """
    def __init__(self, openai_api_key, persona, base_url="https://openai.vocareum.com/v1", model="gpt-4.1-nano"):
        """
        Initializes the AugmentedPromptAgent with API key, persona, base URL, and model.

        Args:
            openai_api_key (str): The API key for OpenAI.
            persona (str): The persona description for the agent.
            base_url (str, optional): The base URL for the OpenAI API.
                                      Defaults to "https://openai.vocareum.com/v1".
            model (str, optional): The model to use for the API calls.
                                   Defaults to "gpt-4.1-nano".
        """
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.base_url = base_url
        self.model = model

    def respond(self, user_prompt):
        """Generate a response using OpenAI API."""
        client = OpenAI(
            api_key=self.openai_api_key,
            base_url=self.base_url
            )

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": f"The persona given to you is: {self.persona}. Forget previous context."},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip()

# KnowledgeAugmentedPromptAgent class definition
class KnowledgeAugmentedPromptAgent:
    """
    An agent that uses a persona and a knowledge base to generate responses. It only consults the knowledge base for answers.
    """
    def __init__(self, openai_api_key, persona, knowledge, base_url="https://openai.vocareum.com/v1", model="gpt-4.1-nano"):
        """
        Initializes the KnowledgeAugmentedPromptAgent with API key, persona, knowledge, base URL, and model.

        Args:
            openai_api_key (str): The API key for OpenAI.
            persona (str): The persona description for the agent.
            knowledge (str): The knowledge base to use for generating responses.
            base_url (str, optional): The base URL for the OpenAI API.
                                      Defaults to "https://openai.vocareum.com/v1".
            model (str, optional): The model to use for the API calls.
                                   Defaults to "gpt-4.1-nano".
        """
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.knowledge = knowledge
        self.base_url = base_url
        self.model = model

    def respond(self, user_prompt):
        """Generate a response using the OpenAI API."""
        client = OpenAI(
            api_key=self.openai_api_key,
            base_url=self.base_url
            )

        system_prompt = f"""
        The persona given to you is: {self.persona}. Forget previous context.\n
        Use only the following knowledge to answer, do not use your own knowledge: {self.knowledge}\n
        Answer the prompt based on this knowledge, not your own."
        """
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )
        return response.choices[0].message.content

# RAGKnowledgePromptAgent class definition
class RAGKnowledgePromptAgent:
    """
    An agent that uses Retrieval-Augmented Generation (RAG) to find knowledge from a large corpus
    and leverages embeddings to respond to prompts based solely on retrieved information.
    """

    def __init__(self, openai_api_key, persona, chunk_size=2000, chunk_overlap=100, base_url="https://openai.vocareum.com/v1", model="gpt-4.1-nano"):
        """
        Initializes the RAGKnowledgePromptAgent with API credentials and configuration settings.

        Parameters:
            openai_api_key (str): The API key for OpenAI.
            persona (str): The persona description for the agent.
            chunk_size (int): The size of text chunks for embedding. Defaults to 2000.
            chunk_overlap (int): Overlap between consecutive chunks. Defaults to 100.
            base_url (str, optional): The base URL for the OpenAI API.
                                      Defaults to "https://openai.vocareum.com/v1".
            model (str, optional): The model to use for the API calls.
                                   Defaults to "gpt-4.1-nano".
        """
        self.persona = persona
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.openai_api_key = openai_api_key
        self.model = model
        self.base_url = base_url
        self.unique_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.csv"

    def get_embedding(self, text):
        """
        Fetches the embedding vector for given text using OpenAI's embedding API.

        Parameters:
        text (str): Text to embed.

        Returns:
        list: The embedding vector.
        """
        client = OpenAI(base_url=self.base_url, api_key=self.openai_api_key)
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding

    def calculate_similarity(self, vector_one, vector_two):
        """
        Calculates cosine similarity between two vectors.

        Parameters:
        vector_one (list): First embedding vector.
        vector_two (list): Second embedding vector.

        Returns:
        float: Cosine similarity between vectors.
        """
        vec1, vec2 = np.array(vector_one), np.array(vector_two)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def chunk_text(self, text):
        """
        Splits text into manageable chunks, attempting natural breaks.

        Parameters:
        text (str): Text to split into chunks.

        Returns:
        list: List of dictionaries containing chunk metadata.
        """
        separator = "\n"
        text = re.sub(r'\s+', ' ', text).strip()

        if len(text) <= self.chunk_size:
            return [{"chunk_id": 0, "text": text, "chunk_size": len(text)}]

        chunks, start, chunk_id = [], self.chunk_overlap, 0

        while start < len(text):
            start = start - self.chunk_overlap
            end = min(start + self.chunk_size, len(text))
            if separator in text[start:end]:
                end = start + text[start:end].rindex(separator) + len(separator)

            chunks.append({
                "chunk_id": chunk_id,
                "text": text[start:end],
                "chunk_size": end - start,
                "start_char": start,
                "end_char": end
            })
            start = end
            chunk_id += 1

        with open(f"chunks-{self.unique_filename}", 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["text", "chunk_size"])
            writer.writeheader()
            for chunk in chunks:
                writer.writerow({k: chunk[k] for k in ["text", "chunk_size"]})

        return chunks

    def calculate_embeddings(self):
        """
        Calculates embeddings for each chunk and stores them in a CSV file.

        Returns:
        DataFrame: DataFrame containing text chunks and their embeddings.
        """
        # Assuming chunks are already calculated. Improvement: check if file exists and if not: chunk text (but: needs text to chunk)
        df = pd.read_csv(f"chunks-{self.unique_filename}", encoding='utf-8')
        df['embeddings'] = df['text'].apply(self.get_embedding)
        df.to_csv(f"embeddings-{self.unique_filename}", encoding='utf-8', index=False)
        return df

    def find_prompt_in_knowledge(self, prompt):
        """
        Finds and responds to a prompt based on similarity with embedded knowledge.

        Parameters:
        prompt (str): User input prompt.

        Returns:
        str: Response derived from the most similar chunk in knowledge.
        """
        prompt_embedding = self.get_embedding(prompt)
        # Assuming embeddings are already calculated. Improvement: check if file exists and if not: generate embeddings.
        df = pd.read_csv(f"embeddings-{self.unique_filename}", encoding='utf-8')
        df['embeddings'] = df['embeddings'].apply(lambda x: np.array(eval(x)))
        df['similarity'] = df['embeddings'].apply(lambda emb: self.calculate_similarity(prompt_embedding, emb))

        best_chunk = df.loc[df['similarity'].idxmax(), 'text']

        client = OpenAI(base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": f"The persona given to you is: {self.persona}. Forget previous context."},
                {"role": "user", "content": f"Answer based only on this information: {best_chunk}. Prompt: {prompt}"}
            ],
            temperature=0
        )

        return response.choices[0].message.content
    
    def respond(self, prompt):
        """
        Alias method for self.find_prompt_in_knowledge to allow the agent to be used like a worker agent.
        """
        return self.find_prompt_in_knowledge(prompt)

class EvaluationAgent:
    """
    An agent that evaluates responses from another agent based on predefined criteria. Worker agents' responses are iteratively refined until they meet the evaluation criteria or the maximum number of interactions is reached.
    """
    
    def __init__(self, openai_api_key, persona, evaluation_criteria, worker_agent, max_interactions, base_url="https://openai.vocareum.com/v1", model="gpt-4.1-nano"):
        """
        Initializes the EvaluationAgent with API credentials and configuration settings.

        Parameters:
            openai_api_key (str): The API key for OpenAI.
            persona (str): The persona description for the agent.
            evaluation_criteria (str): The criteria against which responses will be evaluated.
            worker_agent (Agent object from base_agents.py): The agent that generates responses to be evaluated.
            max_interactions (int): The maximum number of interactions allowed for evaluation.
            base_url (str, optional): The base URL for the OpenAI API.
                                      Defaults to "https://openai.vocareum.com/v1".
            model (str, optional): The model to use for the API calls.
                                   Defaults to "gpt-4.1-nano".
        """
        self.persona = persona
        self.openai_api_key = openai_api_key
        self.model = model
        self.base_url = base_url
        self.evaluation_criteria = evaluation_criteria
        self.worker_agent = worker_agent
        self.max_interactions = max_interactions

    def evaluate(self, initial_prompt):
        """
        Tries to create a valid response to an initial prompt by iteratively refining the worke agent's response based on the evaluation agent's evaluation criteria.
        Parameters: 
            initial_prompt (str): The initial user prompt to evaluate.
        Returns: 
             dict: A dictionary containing the final response ('final_response'), evaluation ('evaluation'), and number of iterations ('num_iterations').
        """

        client = OpenAI(base_url=self.base_url, api_key=self.openai_api_key)
        prompt_to_evaluate = initial_prompt

        for i in range(0,self.max_interactions):
            print(f"\n--- Interaction {i+1} ---")

            print(" Step 1: Worker agent generates a response to the prompt")
            print(f"Prompt:\n{prompt_to_evaluate}")
            response_from_worker = self.worker_agent.respond(prompt_to_evaluate)
            print(f"Worker Agent Response:\n{response_from_worker}")

            print(" Step 2: Evaluator agent judges the response")
            eval_prompt = (
                f"Does the following answer: {response_from_worker}\n"
                f"Meet this criteria: {self.evaluation_criteria}?"  
                f"Respond Yes or No, and the reason why it does or doesn't meet the criteria."
            )
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"The persona given to you is: {self.persona}. Forget previous context."},
                    {"role": "user", "content": f"{eval_prompt}"}
                ],
                temperature=0
            )
            evaluation = response.choices[0].message.content.strip()
            print(f"Evaluator Agent Evaluation:\n{evaluation}")

            print(" Step 3: Check if evaluation is positive")
            if evaluation.lower().startswith("yes"):
                print("✅ Final solution accepted.")
                break
            else:
                print(" Step 4: Generate instructions to correct the response")
                instruction_prompt = (
                    f"Provide instructions to fix an answer based on these reasons why it is incorrect: {evaluation}"
                )
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": f"The persona given to you is: {self.persona}. Forget previous context."},
                        {"role": "user", "content": f"{instruction_prompt}"}
                    ],
                    temperature=0
                )
                instructions = response.choices[0].message.content.strip()
                print(f"Instructions to fix:\n{instructions}")

                print(" Step 5: Send feedback to worker agent for refinement")
                prompt_to_evaluate = (
                    f"The original prompt was: {initial_prompt}\n"
                    f"The response to that prompt was: {response_from_worker}\n"
                    f"It has been evaluated as incorrect.\n"
                    f"Make only these corrections, do not alter content validity: {instructions}"
                )
        return {
            # TODO: 7 - Return a dictionary containing the final response, evaluation, and number of iterations - DONE, left here for reference
            'final_response': response_from_worker,
            'evaluation': evaluation,
            'num_iterations': i + 1
        }

    def respond(self, prompt):
        """
        Alias method for self.evaluate to allow the agent to be used like a worker agent.

        Parameters:
            prompt (str): The user input prompt to evaluate.

        Returns:
            dict: A dictionary containing the final response ('final_response'), evaluation ('evaluation'), and number of iterations ('num_iterations').
        """
        
        return self.evaluate(prompt)

class RoutingAgent():
    """
    An agent that plans which agent to send a user prompt to. 
    """
    def __init__(self, openai_api_key, agents, base_url="https://openai.vocareum.com/v1", model="gpt-4.1-nano"):
        """
        Initializes the RoutingAgent with API credentials and configuration settings.

        Parameters:
            openai_api_key (str): The API key for OpenAI.
            agents (list): A list of agent objects that the router can choose from. Each agent should be a dictionary with keys "name", "description", and "func" (the function to call for that agent).
            base_url (str, optional): The base URL for the OpenAI API.
                                      Defaults to "https://openai.vocareum.com/v1".
            model (str, optional): The model to use for the API calls.
                                   Defaults to "gpt-4.1-nano".
        """
        # Initialize the agent with given attributes
        self.openai_api_key = openai_api_key
        self.agents = agents
        self.model = model
        self.base_url = base_url

    def get_embedding(self, text):
        """
        Fetches the embedding vector for given text using OpenAI's embedding API.

        Parameters:
        text (str): Text to embed.

        Returns:
        list: The embedding vector.
        """
        client = OpenAI(base_url=self.base_url, api_key=self.openai_api_key)
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float"
        )

        embedding = response.data[0].embedding
        return embedding 

    # TODO: 3 - Define a method to route user prompts to the appropriate agent
    def route(self, user_input):
        """
        Routes the user input to the most suitable agent based on the similarity of the input to the agent descriptions.
        Parameters:
            user_input (str): The user input prompt to route.
        Returns:
            str: The response from the selected agent.
        """
        # TODO: 4 - Compute the embedding of the user input prompt
        input_emb = self.get_embedding(user_input)
        best_agent = None
        best_score = -1

        for agent in self.agents:
            # TODO: 5 - Compute the embedding of the agent description
            agent_emb = self.get_embedding(agent["name"] + ": " + agent["description"])

            if agent_emb is None:
                continue

            similarity = np.dot(input_emb, agent_emb) / (np.linalg.norm(input_emb) * np.linalg.norm(agent_emb))
            # print(similarity) # Commented out since it clutters the output, but can be useful for debugging

            # TODO: 6 - Add logic to select the best agent based on the similarity score between the user prompt and the agent descriptions
            if similarity > best_score:
                best_score = similarity
                best_agent = agent

        if best_agent is None:
            return "Sorry, no suitable agent could be selected."

        print(f"[Router] Best agent: {best_agent['name']} (score={best_score:.3f})")
        return best_agent["func"](user_input)


'''
class ActionPlanningAgent:

    def __init__(self, openai_api_key, knowledge):
        # TODO: 1 - Initialize the agent attributes here

    def extract_steps_from_prompt(self, prompt):

        # TODO: 2 - Instantiate the OpenAI client using the provided API key
        # TODO: 3 - Call the OpenAI API to get a response from the "gpt-3.5-turbo" model.
        # Provide the following system prompt along with the user's prompt:
        # "You are an action planning agent. Using your knowledge, you extract from the user prompt the steps requested to complete the action the user is asking for. You return the steps as a list. Only return the steps in your knowledge. Forget any previous context. This is your knowledge: {pass the knowledge here}"

        response_text = ""  # TODO: 4 - Extract the response text from the OpenAI API response

        # TODO: 5 - Clean and format the extracted steps by removing empty lines and unwanted text
        steps = response_text.split("\n")

        return steps
'''