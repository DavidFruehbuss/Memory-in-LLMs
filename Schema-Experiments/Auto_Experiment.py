# This skript implements the interaction of an agent with the Schema-Engine

import os
import random
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from Schema_Engine import LocationPattern

token = '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'
model_id = "meta-llama/Llama-2-70b-chat-hf" 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

def setup_model(model_name, local_dir="/scratch-local/dfruhbus/model_data", device=device):
    
    local_model_path = os.path.join(local_dir, model_name)
    local_tokenizer_path = os.path.join(local_dir, "tokenizer", model_name)

    if not os.path.exists(local_model_path):
        os.makedirs(local_model_path, exist_ok=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=token).to(device)

        model.save_pretrained(local_model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(local_model_path, use_auth_token=token).to(device)


    if not os.path.exists(local_tokenizer_path):
        os.makedirs(local_tokenizer_path, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
        tokenizer.save_pretrained(local_tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path, use_auth_token=token)

    return tokenizer, model

def generate_with_model(prompt, tokenizer, model):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    inputs.to(device)
    outputs = model.generate(inputs, max_length=50)  # Adjust max_length as needed
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def run_episode_with_llm(num_actions, model_name, device=device):
    tokenizer, model = setup_model(model_name, device)
    pattern = LocationPattern()
    previous_feedback = None
    results = []

    for i in range(num_actions):
        prompt = generate_prompt(previous_feedback, is_first_prompt=(i == 0))
        response = generate_with_model(prompt, tokenizer, model)
        action = interpret_response(response)
        action = action.cpu()

        previous_feedback = pattern.provide_feedback(action)
        results.append((action, previous_feedback))

    return results

# Example usage
model_name = model_id  # Replace with the LLaMA model name you're using
episode = run_episode_with_llm(10, model_name, device)
for action, feedback in episode:
    print(f"Action: {action}, Feedback: {feedback}")
