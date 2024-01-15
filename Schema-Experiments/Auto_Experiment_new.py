# This skript implements the interaction of an agent with the Schema-Engine

import os
import random
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

from Schema_Engine import LocationPattern

token = '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'
model_id = "meta-llama/Llama-2-70b-chat-hf" 

# Set CUDA environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
accelerator = Accelerator()
print("Available devices:", accelerator.device)
print("Number of GPUs available:", torch.cuda.device_count())

def setup_model(model_name, local_dir="/scratch-local/dfruhbus/model_data", device=device):
    
    local_model_path = os.path.join(local_dir, model_name)
    local_tokenizer_path = os.path.join(local_dir, "tokenizer", model_name)

    # if not os.path.exists(local_model_path):
    #     os.makedirs(local_model_path, exist_ok=True)
    #     model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=token)

    #     model.save_pretrained(local_model_path)
    # else:
    #     model = AutoModelForCausalLM.from_pretrained(local_model_path, use_auth_token=token)


    # if not os.path.exists(local_tokenizer_path):
    #     os.makedirs(local_tokenizer_path, exist_ok=True)
    #     tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
    #     tokenizer.save_pretrained(local_tokenizer_path)
    # else:
    #     tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path, use_auth_token=token)

    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=token, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)

    model = accelerator.prepare(model)

    return tokenizer, model

def generate_prompt(previous_moves, previous_feedback, is_first_prompt=False):
    '''
    Generate a prompt for a language model, keeping track of the conversation history.

    :param previous_moves: List of tuples containing previous user moves and model responses.
                           Each tuple is in the format (user_msg, model_response).
    :param previous_feedback: Feedback from the last move (e.g., "no reward" or "reward").
    :param is_first_prompt: Boolean indicating if this is the first prompt in the conversation.
    :return: A formatted prompt string for the language model.
    '''
    task_explanation = ''' You are trying to find rewards in an environment with 4 locations A, B, C, D.
    Each round you can choose a location and get feedback whether you received a reward at that location or not.
    Try to find as many rewards as possible. Always answer with only the letter of the location that you would like to visit! Don't give an explanation! '''

    # Initial system message
    prompt = "<s>[INST] <<SYS>>\n"
    prompt += "{}\n<</SYS>>\n".format(task_explanation)
    base_prompt_len = len(prompt)

    # Add conversation history
    for user_msg, model_response in previous_moves:
        # base_prompt does not need to be in history
        prompt += "{{ {} }} [/INST] {{ {} }} </s><s>[INST] ".format(user_msg, model_response)

    # Add current move
    user_input = ""
    if is_first_prompt:
        user_input += "This is your first move. Where do you go?"
    else:
        user_input += "Your previous move received: {}. Where do you go next?".format(previous_feedback)

    user_input += " [/INST]</s>"

    prompt += user_input

    return prompt, user_input


def interpret_response(llm_response):
    # This function should be tailored to the expected format of the LLM response.
    # Assuming the LLM response is simply the chosen location like 'A', 'B', 'C', or 'D'.
    # Assumes string and that answer is only uppercased letter that stands alone
    import re

    # Regular expression to find single uppercase letters that stand alone
    match = re.search(r'\b[A-Z]\b', llm_response)
    return match.group(0) if match else None

def generate_with_model(prompt, tokenizer, model):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda:0")
    outputs = model.generate(inputs, max_length=2000)  # Adjust max_length as needed
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def run_episode_with_llm(num_actions):
    tokenizer, model = setup_model(model_id)
    pattern = LocationPattern()
    previous_feedback = None
    conversation_history = []
    results = []

    for i in range(num_actions):
        prompt, user_input = generate_prompt(conversation_history, previous_feedback, is_first_prompt=(i == 0))
        response = generate_with_model(prompt, tokenizer, model)
        # cut out the promt from the answer:
        response = response.split("[/INST]")[-1]
        print(f'Response: {response}')
        # add promt-answer pair to the conversation history
        conversation_history += [(user_input, response)]
        action = interpret_response(response)

        previous_feedback = pattern.provide_feedback(action)
        results.append((action, previous_feedback))

    return results

def generate_prompt_compressed(results, is_first_prompt=False):
    '''
    Generate a prompt for a language model, keeping track of the conversation history.

    :param previous_moves: List of tuples containing previous user moves and model responses.
                           Each tuple is in the format (user_msg, model_response).
    :param previous_feedback: Feedback from the last move (e.g., "no reward" or "reward").
    :param is_first_prompt: Boolean indicating if this is the first prompt in the conversation.
    :return: A formatted prompt string for the language model.
    '''
    task_explanation = ''' You are trying to find rewards in an environment with 4 locations A, B, C, D.
    Each round you can choose a location and get feedback whether you received a reward at that location or not.
    Try to find as many rewards as possible. Always answer with only the letter of the location that you would like to visit! Don't give an explanation! '''

    # Initial system message
    prompt = "<s>[INST] <<SYS>>\n"
    prompt += "{}\n<</SYS>>\n".format(task_explanation)
    base_prompt_len = len(prompt)

    previous_feedback_list = "["
    for i, (action,reward) in enumerate(results):
      previous_feedback_list += f'Round {i+1}: (Action: {action}, Reward: {reward}),'
    previous_feedback_list += "]"


    # Add current move
    user_input = ""
    if is_first_prompt:
        user_input += "This is your first move. Where do you go?"
    else:
        user_input += "This is a history of your previous actions and the reward you received for taking this action: {}. Where do you go next?".format(previous_feedback_list)

    user_input += " [/INST]</s>"

    prompt += user_input

    return prompt, user_input

def run_episode_with_llm_compressed(num_actions):
    tokenizer, model = setup_model(model_id)
    pattern = LocationPattern()
    previous_feedback = None
    results = []

    for i in range(num_actions):
        prompt, user_input = generate_prompt_compressed(results, is_first_prompt=(i == 0))
        print(f'Prompt: {prompt}')
        response = generate_with_model(prompt, tokenizer, model)
        # cut out the promt from the answer:
        response = response.split("[/INST]")[-1]
        print(f'Response: {response}')
        # add promt-answer pair to the conversation history
        action = interpret_response(response)

        previous_feedback = pattern.provide_feedback(action)
        results.append((action, previous_feedback))

    return results

# Example usage
episode = run_episode_with_llm(10)
reward_total = 0
for action, feedback in episode:
    print(f"Action: {action}, Feedback: {feedback}")
    reward_total += feedback
print(f'Reward total is: {reward_total}')

# Example usage
episode = run_episode_with_llm_compressed(10)
reward_total = 0
for action, feedback in episode:
    print(f"Action: {action}, Feedback: {feedback}")
    reward_total += feedback
print(f'Reward total is: {reward_total}')