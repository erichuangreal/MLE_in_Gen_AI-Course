import openai

# Initialize the OpenAI client
client = openai.OpenAI(api_key='')

def get_completion(prompt, model="gpt-4o-mini"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
    )
    return response.choices[0].message.content


# Example usage
text = "What is the capital of canada"
prompt = f"Please answer the following question in one sentence: {text}"
response = get_completion(prompt)
print(response)

text = """
Artificial intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn. It has applications in various fields, including healthcare, finance, and transportation.
"""

prompt = f"Summarize the following text in 6 words:\n{text}"
response = get_completion(prompt)
print(response)

# system prompt
def get_completion_with_system_prompt(system_prompt, user_prompt, model="gpt-4o-mini"):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": "George Washington was the first president of the United States."},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content

# Define the system and user prompts
system_prompt = "You are an inexperienced assistant that speaks with hesitation."
user_prompt = "Can you explain the importance of data privacy in two sentences?"

response = get_completion_with_system_prompt(system_prompt, user_prompt)
print(response)