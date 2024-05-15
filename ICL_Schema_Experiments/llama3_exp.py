import transformers
import torch
import accelerate
import bitsandbytes
# from accelerate import Accelerator

import random
import pickle

random.seed(42)
# accelerator = Accelerator()

RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
BLUE = '\033[94m'
RESET = '\033[0m'

def load_llama3(quantized=False):

    # 16 GB GPU memory without quantization (works super slow on T4)
    # 4 bit quantization is fast
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    if quantized:
      pipeline = transformers.pipeline(
          "text-generation",
          model=model_id,
          model_kwargs={
              "torch_dtype": torch.float16,
              "quantization_config": {"load_in_4bit": True},
              "low_cpu_mem_usage": True,
          },
          device_map="auto",
      )
    else:
      pipeline = transformers.pipeline(
          "text-generation",
          model=model_id,
          model_kwargs={"torch_dtype": torch.bfloat16},
          device_map="auto",
      )

    # model = accelerator.prepare(model) # pipeline

    return pipeline

def query_llama3(prompt, system_prompt, pipeline): 

    messages = [
        {"role": "system", "content": f"{system_prompt}"},
        {"role": "user", "content": f"{prompt}"},
    ]

    # inputs = accelerator.prepare(inputs) # messages

    prompt = pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        prompt,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    return outputs[0]["generated_text"][len(prompt):]

def generate_prompt(sample, num_examples):

    element, info = sample
    sequences_pos, sequences_neg = element

    system_prompt = 'You are a helpful AI assistant.'
    prompt = 'You are supposed to identify whether a list of letters entails an unknown pattern based on examples.\n'

    prompt_example_list = []

    pos_examples = random.sample(sequences_pos, num_examples + 1)
    for i in range(1, len(pos_examples)):
      prompt_example_list += [f'{pos_examples[i][0]} \nThe correct answer is: {pos_examples[i][1]} \n']

    neg_examples = random.sample(sequences_neg, num_examples + 1)
    for i in range(1, len(neg_examples)):
      prompt_example_list += [f'{neg_examples[i][0]}  \nThe correct answer is: {neg_examples[i][1]} \n']

    random.shuffle(prompt_example_list)
    for example in prompt_example_list:
      prompt += example

    new_seqeunce = random.sample([pos_examples[0], neg_examples[0]], 1)[0]
    prompt += f'{new_seqeunce[0]} \nThe correct answer is: '
    corrent_label = new_seqeunce[1]

    return prompt, system_prompt, corrent_label

def check_answer(input_string):

    # Check for 1
    if '1' in input_string:
        return 1
    # Check for 0
    elif '0' in input_string:
        return 0
    else:
        return None
    
def single_query(sample, num_examples, show_example_text=False):

    prompt, system_prompt, corrent_label = generate_prompt(sample, num_examples)

    if show_example_text:
        print(prompt)
        print(f'{RED}Correct Label is: {corrent_label} \n{RESET}')

    answer = query_llama3(prompt, system_prompt, pipeline)
    predicted_label = check_answer(answer)

    if show_example_text:
        print(f'{RED}Predicted Label is: {predicted_label} \n{RESET}')
        print(f'{BLUE}Answer:\n {answer}{RESET} \n')

    if predicted_label == corrent_label:
      return 1, (corrent_label)
    else:
      return 0, (corrent_label)
    
def experiment(repetitions=10, dataset):

    results = {}

    for i, sample in enumerate(dataset):

        results[i] = {}

        for num_examples in range(1, 6):

            results[i][num_examples] = {}
            results[i][num_examples]['pos'] = []
            results[i][num_examples]['neg'] = []

            for j in range(repetitions):

                if i == 0 and j == 0:
                    show_example_text=True
                else:
                    show_example_text=False

                score, corrent_label = single_query(sample, num_examples, show_example_text)

                if corrent_label == 0:
                    results[i][num_examples]['neg'] += [score]
                else:
                    results[i][num_examples]['pos'] += [score]

    return results

def show_results(results, pattern_type='all', n_examples='all', lengths='all', vocab_size='all'):

    total_score = 0
    max_score = 0

    p = 'pos'
    n = 'neg'

    print('Showing:')
    settings = f'pattern_type={pattern_type}, n_examples={n_examples}, lengths={lengths}, vocab_size={vocab_size}'
    print(f'{settings} \n')

    for i, sample in enumerate(dataset):

        if pattern_type != 'all' and dataset[i][1][0] != pattern_type:
            continue
        if lengths != 'all' and dataset[i][1][1] != lengths:
            continue
        if vocab_size != 'all' and dataset[i][1][2] != vocab_size:
          continue

        element, info = sample

        print(f'{GREEN}{info}: \n{RESET}')

        for num_examples in range(1, 6):

            if n_examples != 'all' and num_examples != n_examples:
                continue

            total_score += sum(results[i][num_examples][p]) + sum(results[i][num_examples][n])
            max_score += len(results[i][num_examples][p]) + len(results[i][num_examples][n])

            print(f'Number of examples: {num_examples}')
            print(f'Score for sequences with pattern: {results[i][num_examples][p]}')
            print(f'Score for sequences without pattern: {results[i][num_examples][n]}')

        print('')

    return total_score, max_score, settings


# main function
if __name__ == '__main__':

    """
    Need modules installed in the env and need to be logged in to huggingface
    """

    file_path_dataset = './icl_llama3_dataset/dataset_2.pkl'

    with open(file_path_dataset, 'rb') as f:
        dataset = pickle.load(f)

    pipeline = load_llama3(True)

    compute_new = True

    file_path_save = './icl_llama3_results/results_Llama3_test.pkl'

    if compute_new == True:

        # Testing
        results = experiment(1, dataset=dataset[:10]) # currently set to 10/240 

        with open(file_path_save, 'wb') as f:
            pickle.dump(results, f)

    else:

        with open(file_path_save, 'rb') as f:
            results = pickle.load(f)


