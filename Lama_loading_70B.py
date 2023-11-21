from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

if __name__ == '__main__':
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # model = transformers.AutoModelForCausalLM.from_pretrained(
    #     'meta-llama/Llama-2-7b-chat-hf'
    # )

    token='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'

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-chat-hf", use_auth_token=token)
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-70b-chat-hf", use_auth_token=token).to(device)

    model.eval()
    model.to(device)

    print(model)