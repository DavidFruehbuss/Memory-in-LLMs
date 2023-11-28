# This skript implements the interaction of an agent with the Schema-Engine with GPT4

from openai import OpenAI

from Schema_Engine import LocationPattern

api_key = "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"

def generate_prompt(previous_feedback, is_first_prompt=False):
    ''' Currently I am assuming that the history of the conversation is automatically added '''
    task_explanation = ''' You are trying to find rewards in an environment with 4 locations A, B, C, D. 
    Each round you can choose a location and get feedback whether you received a reward at that location or not. 
    Try to find as many rewards as possible. '''
    if is_first_prompt:
        return f"{task_explanation} This is your first move. Where do you go?"
    else:
        return f"Your previous move received: {previous_feedback}. Where do you go next?"

def interpret_response(llm_response):
    # This function should be tailored to the expected format of the LLM response.
    # Assuming the LLM response is simply the chosen location like 'A', 'B', 'C', or 'D'.
    print(llm_response)
    return llm_response.strip()

def query_gpt(prompt, api_key=api_key, model="gpt-4"):
    client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=api_key,
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": ''' You are trying to find rewards in an environment with 4 locations A, B, C, D. 
                Each round you can choose a location and get feedback whether you received a reward at that location or not. 
                Try to find as many rewards as possible. '''},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message

def run_episode_with_llm(num_actions):
    pattern = LocationPattern()
    previous_feedback = None
    results = []
    conversation = ""

    for i in range(num_actions):
        prompt = generate_prompt(previous_feedback, is_first_prompt=(i == 0))
        conversation += f"\nHuman: {prompt}\nAI:"
        response = query_gpt(conversation)
        print("AI:", response)
        conversation += response
        action = interpret_response(response)

        previous_feedback = pattern.provide_feedback(action)
        results.append((action, previous_feedback))

    return results

# Example usage
episode = run_episode_with_llm(10)
for action, feedback in episode:
    print(f"Action: {action}, Feedback: {feedback}")
