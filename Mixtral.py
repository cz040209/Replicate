import requests
import streamlit as st

# Set up API Key
api_key = st.secrets["groq_api"]["api_key"]

# Base URL and headers
base_url = "https://api.groq.com/openai/v1"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Model ID
model_id = "mixtral-8x7b-32768"

# Step 1: Verify Model (Optional)
model_url = f"{base_url}/models/{model_id}"

try:
    model_response = requests.get(model_url, headers=headers)
    if model_response.status_code == 200:
        print("Model verified successfully.")
    else:
        print(f"Error {model_response.status_code}: {model_response.text}")
        st.stop()  # Stop execution if model is invalid
except requests.exceptions.RequestException as e:
    print(f"An error occurred while verifying the model: {e}")
    st.stop()

# Step 2: Use the Model for Chat Completions
url = f"{base_url}/chat/completions"
data = {
    "model": model_id,
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    "temperature": 0.7,
    "max_tokens": 200,
    "top_p": 0.9
}

try:
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        bot_response = result['choices'][0]['message']['content']
        print(f"Bot: {bot_response}")
    else:
        print(f"Error {response.status_code}: {response.text}")
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
