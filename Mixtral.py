import requests
import streamlit as st
import PyPDF2

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

# Custom CSS for styling
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
        .botify-title {
            font-family: 'Arial', sans-serif';
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

# Initialize interaction history in session state
if "history" not in st.session_state:
    st.session_state.history = []
if "selected_interaction" not in st.session_state:
    st.session_state.selected_interaction = None

# Step 1: Function to Extract Text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    extracted_text = ""
    for page in pdf_reader.pages:
        extracted_text += page.extract_text()
    return extracted_text

# Sidebar for interaction history
st.sidebar.header("Interaction History")
if st.session_state.history:
    interaction_options = [f"Interaction {i + 1}" for i in range(len(st.session_state.history))]
    selected_index = st.sidebar.selectbox(
        "Select an interaction to view details:",
        options=interaction_options
    )
    if selected_index:
        index = int(selected_index.split(" ")[1]) - 1
        st.session_state.selected_interaction = st.session_state.history[index]

# Display selected interaction in the sidebar
if st.session_state.selected_interaction:
    st.sidebar.subheader("Selected Interaction")
    st.sidebar.write(f"**Question:** {st.session_state.selected_interaction['question']}")
    st.sidebar.write(f"**Response:** {st.session_state.selected_interaction['response']}")

# Model selection
selected_model_name = st.selectbox("Choose a model:", list(available_models.keys()))
selected_model_id = available_models[selected_model_name]

# File upload
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    # Extract text from the uploaded PDF
    st.write("Extracting text from the uploaded PDF...")
    pdf_text = extract_text_from_pdf(uploaded_file)
    st.success("Text extracted successfully!")

    # Display extracted text (Optional)
    with st.expander("View Extracted Text"):
        st.write(pdf_text)

    # Step 2: User Input for Questions
    question = st.text_input("Ask a question about the PDF content:")

    if question:
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
                st.write(f"Bot ({selected_model_name}): {bot_response}")
                
                # Save interaction to history
                st.session_state.history.append({
                    "question": question,
                    "response": bot_response
                })
            else:
                st.error(f"Error {response.status_code}: {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"An error occurred: {e}")
