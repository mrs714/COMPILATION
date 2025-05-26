from openai import OpenAI
import random

from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

def get_models():
    """
    Retrieves a list of available models from the OpenAI API.
    Returns:
    - A list of model names.
    """
    models = client.models.list()
    return [model.id for model in models.data if model.id.startswith("gpt-")]

def get_response(model_name, query, max_tokens=512):
    """
    Generates a response from the specified model for the given query.
    Parameters:
    - model_name: The name of the model to use for generating the response.
    - query: The input query to be sent to the model.
    - max_tokens: The maximum number of tokens to generate in the response.
    Returns:
    - The generated response as a string.
    """
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "system", "content": "You are a helpful assistant for an AI arbitration system."},
                  {"role": "user", "content": str(query)}],
        temperature=0.7,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()

def check_agreement(model_name, leader_response, own_response):
    """
    Checks if the response from the model agrees with the leader's response.
    Parameters:
    - model_name: The name of the model to use for checking agreement.
    - leader_response: The response from the leader model.
    - own_response: The response from the current model.
    Returns:
    - A boolean indicating whether the two responses agree in their main conclusions.
    """

    comparison_prompt = (
        f"Leader's response:\n{leader_response}\n\n"
        f"Model's response:\n{own_response}\n\n"
        "Do the two responses agree in their main conclusions? Reply with 'Yes' or 'No'."
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": comparison_prompt}],
        temperature=0,
        max_tokens=10
    )
    return response.choices[0].message.content.strip().lower().startswith("yes")

def consensus(models, query, threshold, verbose=False):
    """
    Attempts to reach consensus among a list of models for a given query.
    Parameters:
    - models: List of model names to use for generating responses.
    - query: The input query to be sent to the models.
    - threshold: The minimum agreement rate required to consider the consensus successful.
    - verbose: If True, prints detailed information about the process.
    Returns:
    - A tuple containing the leader's response, a boolean indicating whether consensus was reached,
      and the agreement rate.
    """

    leader_model_num = random.randint(0, len(models) - 1)
    leader_model = models[leader_model_num]
    other_models = models[:leader_model_num] + models[leader_model_num + 1:]

    if verbose:
        print(f"Leader model: {leader_model}")
        print(f"Other models: {other_models}")
        print("Generating leader's response...")

    # Generate leader's response
    leader_response = get_response(leader_model, query)

    if verbose:
        print(f"Leader's response: {leader_response}")

    # Generate responses from other models
    agreements = 0
    for model in other_models:
        own_response = get_response(model, query)
        if verbose:
            print(f"{model}'s response: {own_response}")
        if check_agreement(model, leader_response, own_response):
            agreements += 1

    agreement_rate = agreements / len(other_models)
    
    return (leader_response, agreement_rate >= threshold, agreement_rate)

def consensus_until_agreement(models, query, threshold=0.8, verbose=False, max_retries=5):
    """
    Repeatedly attempts to reach consensus until an agreement is reached or max_retries is exhausted.
    Parameters:
    - models: List of model names to use for generating responses.
    - query: The input query to be sent to the models.
    - threshold: The minimum agreement rate required to consider the consensus successful.
    - verbose: If True, prints detailed information about the process.
    - max_retries: The maximum number of attempts to reach consensus.

    Returns:
    - A tuple containing the leader's response and a boolean indicating whether consensus was reached.
    """

    while max_retries > 0:
        max_retries -= 1
        leader_response, agreement, rate = consensus(models, query, threshold, verbose)
        if agreement:
            return (leader_response, True, rate)
        else:
            print("No consensus reached. Retrying...")

    return (leader_response, False, rate)

"""
Example use: 
models = ["gpt-4", "gpt-3.5-turbo", "gpt-4o"]
query = "What are the benefits of using renewable energy sources?"
threshold = 0.7

if consensus(models, query, threshold, verbose=False):
    print("Consensus achieved.")
else:
    print("Consensus not achieved.")
"""