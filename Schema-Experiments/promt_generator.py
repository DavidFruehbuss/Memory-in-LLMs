# This skript implements the interaction of an agent with the Schema-Engine

import random
import requests

from Schema_Engine import LocationPattern

token = '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'
model_id = "meta-llama/Llama-2-70b-chat-hf"

def generate_prompt(previous_feedback, is_first_prompt=False):
    task_explanation = ''' You are trying to find rewards in an environment with 4 locations A, B, C, D. 
    Each round you can choose a location and get feedback whether you received a reward at that location or not. 
    Try to find as many rewards as possible. '''
    if is_first_prompt:
        return f"{task_explanation} This is your first move. Where do you go?"
    else:
        return f"{task_explanation} Your previous move received: {previous_feedback}. Where do you go next?"

def interpret_response(llm_response):
    # This function should be tailored to the expected format of the LLM response.
    # Assuming the LLM response is simply the chosen location like 'A', 'B', 'C', or 'D'.
    print(llm_response)
    return llm_response.strip()

def query_huggingface(payload, model_id, api_token):
    headers = {"Authorization": f"Bearer {api_token}"}
    API_URL = f"https://api-inference.huggingface.co/models/{model_id}"
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def run_episode_with_llm(num_actions, model_id=model_id, api_token=token):
    pattern = LocationPattern()
    previous_feedback = None
    results = []

    for i in range(num_actions):
        prompt = generate_prompt(previous_feedback, is_first_prompt=(i == 0))
        payload = {"inputs": prompt}

        # Get response from Hugging Face API
        response = query_huggingface(payload, model_id, api_token)
        action = interpret_response(response)

        previous_feedback = pattern.provide_feedback(action)
        results.append((action, previous_feedback))

    return results


# Example usage
episode = run_episode_with_llm(10)
for action, feedback in episode:
    print(f"Action: {action}, Feedback: {feedback}")
