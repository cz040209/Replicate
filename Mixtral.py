import requests
import streamlit as st
import PyPDF2

# Custom CSS for a more premium look
st.markdown("""
    <style>
        .css-1d391kg {
            background-color: #1c1f24;  /* Dark background */
            color: white;
            font-family: 'Arial', sans-serif;
        }
        .css-1v0m2ju {
            background-color: #282c34;  /* Slightly lighter background */
        }
        .css-13ya6yb {
            background-color: #61dafb;  /* Button color */
            border-radius: 5px;
            padding: 10px 20px;
            color: white;
            font-size: 16px;
            font-weight: bold;
        }
        .css-10trblm {
            font-size: 18px;
            font-weight: bold;
            color: #282c34;
        }
        .css-3t9iqy {
            color: #61dafb;
            font-size: 20px;
        }
        .botify-title {
            font-family: 'Arial', sans-serif;
            font-size: 48px;
            font-weight: bold;
            color: #61dafb;
            text-align: center;
            margin-top: 50px;
            margin-bottom: 30px;
        }
    </style>
""", unsafe_allow_html=True)

# Botify Title
st.markdown('<h1 class="botify-title">Botify</h1>', unsafe_allow_html=True)

# Set up API Key from secrets
api_key = st.secrets["groq_api"]["api_key"]

# Base URL and headers for Groq API
base_url = "https://api.groq.com/openai/v1"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Available models
available_models = {
    "Mixtral 8x7b": "mixtral-8x7b-32768",
    "Llama 3.1 70b Versatile": "llama-3.1-70b-versatile"
}

# Step 1: Function to Extract Text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    extracted_text = ""
    for page in pdf_reader.pages:
        extracted_text += page.extract_text()
    return extracted_text

# Function to Summarize the Text
def summarize_text(text, model_id):
    url = f"{base_url}/chat/completions"
    data = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Summarize the following text:"},
            {"role": "user", "content": text}
        ],
        "temperature": 0.7,
        "max_tokens": 300,
        "top_p": 0.9
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return f"Error {response.status_code}: {response.text}"
    except requests.exceptions.RequestException as e:
        return f"An error occurred: {e}"

# Streamlit UI
st.title("PDF Question-Answering Chatbot")

# Model selection
selected_model_name = st.selectbox("Choose a model:", list(available_models.keys()))
selected_model_id = available_models[selected_model_name]

# Sidebar for interaction history
if "history" not in st.session_state:
    st.session_state.history = []

# File upload
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    # Extract text from the uploaded PDF
    st.write("Extracting text from the uploaded PDF...")
    pdf_text = extract_text_from_pdf(uploaded_file)
    st.success("Text extracted successfully!")

    # Display extracted text with adjusted font size
    with st.expander("View Extracted Text"):
        st.markdown(f"<div style='font-size: 14px;'>{pdf_text}</div>", unsafe_allow_html=True)

    # Summarize the extracted text
    if st.button("Summarize Text"):
        st.write("Summarizing the text...")
        summary = summarize_text(pdf_text, selected_model_id)
        st.write("Summary:")
        st.write(summary)

    # Step 2: User Input for Questions
    question = st.text_input("Ask a question about the PDF content:")

    if question:
        # Add user question to history
        st.session_state.history.append(f"You: {question}")
        
        # Use the extracted text and user question for Chat Completions
        url = f"{base_url}/chat/completions"
        data = {
            "model": selected_model_id,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Use the following PDF content to answer the user's questions."},
                {"role": "system", "content": pdf_text},
                {"role": "user", "content": question}
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
                st.session_state.history.append(f"Bot ({selected_model_name}): {bot_response}")
                st.write(f"Bot ({selected_model_name}): {bot_response}")
            else:
                st.error(f"Error {response.status_code}: {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"An error occurred: {e}")

# Display interaction history in the sidebar
with st.sidebar:
    st.subheader("Interaction History")
    for message in st.session_state.history:
        st.write(message)
