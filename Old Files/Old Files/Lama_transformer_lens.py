"""
From the transformer_lens lama_demo notebook
"""

# Imports
import circuitsvis as cv
import torch
from jaxtyping import Float, Int

# transformer_lens
import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache

# important for memory saving
torch.set_grad_enabled(False)


# Load the model
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
import os

MODEL_PATH='meta-llama/Llama-2-70b-chat-hf'
token='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'

# Huggingface Model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_auth_token=token)
hf_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, low_cpu_mem_usage=True, use_auth_token=token)

# HookedTransformer model from Huggingface model
model = HookedTransformer.from_pretrained("meta-llama/Llama-2-70b-chat-hf", hf_model=hf_model, device="cpu", fold_ln=False, center_writing_weights=False, center_unembed=False, tokenizer=tokenizer)

model = model.to("cuda" if torch.cuda.is_available() else "cpu")
print(model.generate("The capital of Germany is", max_new_tokens=20, temperature=0))


# Reading from Hooks
llama_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on taskspecific datasets."
llama_tokens = model.to_tokens(llama_text)
llama_logits, llama_cache = model.run_with_cache(llama_tokens, remove_batch_dim=True)

attention_pattern = llama_cache["pattern", 0, "attn"]
llama_str_tokens = model.to_str_tokens(llama_text)

print("Layer 0 Head Attention Patterns:")
cv.attention.attention_patterns(tokens=llama_str_tokens, attention=attention_pattern)


# Writting hooks
layer_to_ablate = 0
head_index_to_ablate = 31

# We define a head ablation hook
# The type annotations are NOT necessary, they're just a useful guide to the reader
# 
def head_ablation_hook(
    value: Float[torch.Tensor, "batch pos head_index d_head"],
    hook: HookPoint
) -> Float[torch.Tensor, "batch pos head_index d_head"]:
    print(f"Shape of the value tensor: {value.shape}")
    value[:, :, head_index_to_ablate, :] = 0.
    return value

original_loss = model(llama_tokens, return_type="loss")
ablated_loss = model.run_with_hooks(
    llama_tokens, 
    return_type="loss", 
    fwd_hooks=[(
        utils.get_act_name("v", layer_to_ablate), 
        head_ablation_hook
        )]
    )
print(f"Original Loss: {original_loss.item():.3f}")
print(f"Ablated Loss: {ablated_loss.item():.3f}")

