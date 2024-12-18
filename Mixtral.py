import requests
import os

# Set up Groq API Key
api_key = "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"  # Replace with your Groq API key
url = "https://api.groq.com/openai/v1/chat/completions"

# Headers for authentication
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Model to use: Mixtral-8x7B-32768
data = {
    "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",  # Specify the Mixtral model
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},  # System message (optional)
        {"role": "user", "content": "What is the capital of France?"}  # User query
    ],
    "temperature": 0.7,  # Controls randomness of the output
    "max_tokens": 200,   # Limits the response length
    "top_p": 0.9         # Nucleus sampling for diversity
}

# Make the POST request to Groq API
response = requests.post(url, headers=headers, json=data)

# Print the response
if response.status_code == 200:
    result = response.json()
    bot_response = result['choices'][0]['message']['content']
    print(f"Bot: {bot_response}")
else:
    print(f"Error {response.status_code}: {response.text}")
