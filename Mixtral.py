import requests
import os

# Set up Groq API Key securely
api_key = os.getenv("GROQ_API_KEY", "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")  # Use environment variable for security

# Endpoint for Groq Chat Completions API
url = "https://api.groq.com/openai/v1/chat/completions"

# Headers for authentication
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Payload: Using Mixtral-8x7B model
data = {
    "model": "mixtral-8x7b-32768",  # Correct model ID as provided in Groq's response
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    "temperature": 0.7,  # Controls randomness
    "max_tokens": 200,   # Max response length
    "top_p": 0.9         # Nucleus sampling for output diversity
}

# Make the POST request to Groq API
try:
    response = requests.post(url, headers=headers, json=data)

    # Check for successful response
    if response.status_code == 200:
        result = response.json()
        bot_response = result['choices'][0]['message']['content']
        print(f"Bot: {bot_response}")
    else:
        # Print error details if response fails
        print(f"Error {response.status_code}: {response.text}")

except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
