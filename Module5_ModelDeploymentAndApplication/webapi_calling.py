import requests
import json


def get_completion():
    response = requests.get(url='http://localhost:8080/health')
    return response


print(get_completion())

def get_completion_v2(prompt):
    headers = {'Content-Type': 'application/json'}
    data = {"prompt": prompt}
    response = requests.post(url='http://localhost:8080/completions', headers=headers, data=json.dumps(data))
    return response.json()['content']


print(get_completion_v2('<|User|>Hello<|Assistant|>'))

def get_completion_v3(prompt):
    headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer no-key'}
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    response = requests.post(url='http://localhost:8080/v1/chat/completions', headers=headers, data=json.dumps(data))
    return response.json()['choices']


print(get_completion_v3('Hello'))

import openai

client = openai.OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="sk-no-key-required"
)

completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "Who are you?"}
    ]
)

print(completion)
print(completion.choices[0].message)
print(completion.choices[0].message.content)