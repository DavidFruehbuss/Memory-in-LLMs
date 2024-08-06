""" 
The goal of the study is to discover complex memories in an LLM.
"""

# Pipline for running a LLM analysis on a server

## Transformer_Lens Imports

import circuitsvis as cv
# Testing that the library works
cv.examples.hello("cv works")

# Import stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
from fancy_einsum import einsum
import tqdm.auto as tqdm
import random
from pathlib import Path
import plotly.express as px
from torch.utils.data import DataLoader

from jaxtyping import Float, Int
from typing import List, Union, Optional
from functools import partial
import copy

import itertools
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import dataclasses
import datasets
from IPython.display import HTML

import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache

torch.set_grad_enabled(False)

def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def line(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.line(utils.to_numpy(tensor), labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs).show(renderer)

## Other imports
from transformers import LlamaForCausalLM, LlamaTokenizer
# import matplotlib.pyplot as plt


## Study A in large LLM

# loading the model
MODEL_PATH='decapoda-research/llama-65b-hf'
token='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'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Huggingface Model (needed for transformer_lens models)
tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH, device=device, use_auth_token=token)
hf_model = LlamaForCausalLM.from_pretrained(MODEL_PATH, device=device, low_cpu_mem_usage=True, use_auth_token=token)

# loading transformer_lens model
model = HookedTransformer.from_pretrained('llama-65b-hf', hf_model=hf_model, device=device, fold_ln=False, center_writing_weights=False, center_unembed=False, tokenizer=tokenizer)

# testing the model works correctly
text_promt = 'Which of these countries has the biggest capital: China, Germany? The answer is: \n'
logits, loss = model(text_promt, return_type='both')
print(f'Logit shape: {logits.shape}, Logits: {logits}, Loss: {loss}')

# Getting an answer
# easiest using generate function and important to set the temperature very low
answer = model.generate(text_promt, max_new_tokens=2, temperature=0)
print(f'Answer: {answer}')

# looking at the logits
# Not the answer but give some hints at internal model state, when it gives an answer.
response_tokens = torch.argmax(logits, dim=2).squeeze(0)
response = model.to_string(response_tokens)
response_2 = model.tokenizer.decode(response_tokens)
print(f'\n Decoding the model logits with decode (same result as to_sting): {response_2}')
print(f'\n Decoding the model logits: {response}')

# first probe: Logit_Lens to see what the model has internally at each state
def logit_lens(representations, unembed_matrix):
  return representations @ unembed_matrix

model_logits, model_cache = model.run_with_cache(text_promt)
unembed_matrix = model.unembed.W_U
# last tokens residual stream after attn or mlp
representations_attn = torch.cat([torch.unsqueeze(model_cache[f"blocks.{layer}.hook_attn_out"].mean(0)[-1], 0) for layer in range(12)], dim=0)
representations_mlp = torch.cat([torch.unsqueeze(model_cache[f"blocks.{layer}.hook_mlp_out"].mean(0)[-1], 0) for layer in range(12)], dim=0)
mlp_lens_objects = logit_lens(representations_mlp, unembed_matrix)
attn_lens_objects = logit_lens(representations_attn, unembed_matrix)
print(mlp_lens_objects.shape)
print(attn_lens_objects.shape)

# looking at city preference
germany_token = model.tokenizer.encode(' Germany')[0]
china_token = model.tokenizer.encode(' China')[0]
# england_token = model.tokenizer.encode(' England')[0]

germany_token_logit_lenses = mlp_lens_objects[:, germany_token]
china_token_logit_lenses = mlp_lens_objects[:, china_token]
# england_token_logit_lenses = mlp_lens_objects[:, england_token]
logit_lens_diff = germany_token_logit_lenses - china_token_logit_lenses
print(logit_lens_diff)

# Somewhat predictive maybe but hard to interpret and ambigious.
# Might want to look at other tokens and individual heads

extra_plots = False

# second probe (can also use logit_lens)
berlin_token_idx = model.tokenizer.encode(' Berlin')[0]
beijing_token_idx = model.tokenizer.encode(' Beijing')[0]
neg_token_idx = model.tokenizer.encode(' Icecream')[0]

mlp_lens_objects_prob = torch.softmax(mlp_lens_objects, dim=1)
attn_lens_objects_prob = torch.softmax(attn_lens_objects, dim=1)

berlin_lens_mlp = mlp_lens_objects_prob[:, berlin_token_idx]
berlin_lens_attn = attn_lens_objects_prob[:, berlin_token_idx]

beijing_lens_mlp = mlp_lens_objects_prob[:, beijing_token_idx]
beijing_lens_attn = attn_lens_objects_prob[:, beijing_token_idx]

print(f"Does the model look at the capitals? Probe looks at all layers but only the last token\n")
print(f"berlin_lens_mlp: {berlin_lens_mlp}, \nberlin_lens_attn: {berlin_lens_attn}, \nbeijing_lens_mlp: {beijing_lens_mlp}, \nbeijing_lens_attn: {beijing_lens_attn}")

print(f"At least for the last token, it doesn't look like it\n")

print(f"Same Analysis for all tokens, but only showing p(capital) > 0.1\n")

# extracting reps. for all tokens and appling logit lens (need to look at layer_norm influence)
representations_attn_all = torch.cat([torch.unsqueeze(model_cache[f"blocks.{layer}.hook_attn_out"].mean(0)[:], 0) for layer in range(12)], dim=0)
representations_mlp_all = torch.cat([torch.unsqueeze(model_cache[f"blocks.{layer}.hook_mlp_out"].mean(0)[:], 0) for layer in range(12)], dim=0)
mlp_lens_objects_all = logit_lens(representations_mlp_all, unembed_matrix)
attn_lens_objects_all = logit_lens(representations_attn_all, unembed_matrix)
# looking at prob of capital memory token being looked at (indirect recall)
mlp_lens_objects_prob_all = torch.softmax(mlp_lens_objects_all, dim=2)
attn_lens_objects_prob_all = torch.softmax(attn_lens_objects_all, dim=2)
berlin_lens_mlp_all = mlp_lens_objects_prob_all[:, :, berlin_token_idx]
berlin_lens_attn_all = attn_lens_objects_prob_all[:, :,berlin_token_idx]
beijing_lens_mlp_all = mlp_lens_objects_prob_all[:, :, beijing_token_idx]
beijing_lens_attn_all = attn_lens_objects_prob_all[:, :, beijing_token_idx]
neg_lens_mlp_all = mlp_lens_objects_prob_all[:, :, neg_token_idx]
neg_lens_attn_all = attn_lens_objects_prob_all[:, :, neg_token_idx]

capitals = ['berlin_mlp', 'berlin_attn', 'beijing_mlp', 'beijing_attn']
catpital_prob = [berlin_lens_mlp_all, berlin_lens_attn_all, beijing_lens_mlp_all, beijing_lens_attn_all]

for i, prob in enumerate(catpital_prob):
  for layer in range(berlin_lens_mlp_all.shape[0]):
    for token in range(berlin_lens_mlp_all.shape[1]):
      if prob[layer, token] > 0.1:
        print(f"P:{prob[layer, token]}, layer: {layer}, token: {token}, capital: {capitals[i]}")

print(f"\nThere is certanly some identifiable recall there, but it is unclear to me how to proceed with this result")

if extra_plots:
  fig, ax = plt.subplots(1, 4, figsize=(20,20))
  for i in range(4):
    ax[i].imshow(catpital_prob[i].T.cpu().numpy(), cmap='viridis', vmin=0, vmax=1)
    ax[i].set_title(capitals[i])
    ax[i].set(xlabel='layer', ylabel='tokens')
  fig.suptitle('Transformer_lens likelyhood for recall of capitals from memory')

  print(f"Ploting same map for other words as a baseline")

  capitals_neg = ['neg_mlp', 'neg_attn']
  catpital_prob_neg = [neg_lens_mlp_all, neg_lens_attn_all]

  fig, ax = plt.subplots(1, 2, figsize=(10,10))
  for i in range(2):
    ax[i].imshow(catpital_prob_neg[i].T.cpu().numpy(), cmap='viridis', vmin=0, vmax=1)
    ax[i].set_title(capitals_neg[i])
    ax[i].set(xlabel='layer', ylabel='tokens')
  fig.suptitle('Transformer_lens likelyhood for recall of other capitals and words from memory')

  print('Token_idxs', berlin_token_idx, beijing_token_idx, neg_token_idx) 