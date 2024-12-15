import os
import streamlit as st
import replicate  # Replicate platform
import speech_recognition as sr  # For audio-to-text functionality
from PIL import Image
import pytesseract  # For OCR (Image to Text)
import fitz  # PyMuPDF for PDF text extraction

# Set the path to the tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\chenz\Downloads\tesseract-ocr-w64-setup-5.5.0.20241111\tesseract.exe'

# Your Replicate API token
REPLICATE_API_TOKEN = "r8_7jHKZQM9VHpRiGGB1HOIQJUqvcidzDz41avpZ"  

# Set up the replicate client
client = replicate.Client(api_token=REPLICATE_API_TOKEN)

# Function to load Summarization model
def load_summarization_model():
    model_name = "facebook/bart-large-cnn"  # Replace with your model from Replicate
    model = client.models.get(model_name)
    return model

# Function to load Translation model
def load_translation_model():
    model_name = "Helsinki-NLP/opus-mt-en-zh"  # Example: English to Chinese
    model = client.models.get(model_name)
    return model

# Function to load Llama 2 model
def load_llama_model():
    model_name = "meta-llama/Llama-2-7b-chat-hf"  # Replace with Llama 2 from Replicate
    model = client.models.get(model_name)
    return model

# Initialize models
summarization_model = load_summarization_model()
translation_model = load_translation_model()
llama_model = load_llama_model()

# Function to split text into manageable chunks for summarization
def split_text(text, max_tokens=1024):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Function to summarize text
def summarize_text(text):
    max_tokens = 1024  # Token limit for the model
    chunks = split_text(text, max_tokens)

    summaries = []
    for chunk in chunks:
        summary = summarization_model.predict(inputs={"text": chunk})
        summaries.append(summary)

    final_summary = " ".join(summaries)
    return final_summary if summaries else "No summary available."

# Function to extract text from PDF using PyMuPDF
def extract_text_from_pdf(pdf_file):
    with fitz.open(pdf_file) as pdf:
        text = ""
        for page in pdf:
            text += page.get_text("text")  # Extract text
    return text

# Function for Audio-to-Text (Speech Recognition)
def audio_to_text(audio_file):
    recognizer = sr.Recognizer()
    audio = sr.AudioFile(audio_file)
    with audio as source:
        audio_data = recognizer.record(source)
    return recognizer.recognize_google(audio_data)

# Function for Image to Text (OCR)
def image_to_text(image_file):
    image = Image.open(image_file)
    text = pytesseract.image_to_string(image)
    return text

# History storage - will store interactions as tuples (user_input, response_output)
if 'history' not in st.session_state:
    st.session_state.history = []

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

# Option to choose between PDF upload, manual input, or translation
option = st.selectbox("Choose input method:", ("Upload PDF", "Enter Text Manually", "Upload Audio", "Upload Image"))

context_text = ""

# Handling different options
if option == "Upload PDF":
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf", label_visibility="collapsed")

    if uploaded_file:
        with st.spinner("Extracting text from PDF..."):
            pdf_text = extract_text_from_pdf(uploaded_file)
        if pdf_text:
            st.success("Text extracted successfully!")
            st.text_area("Extracted Text (Preview)", pdf_text[:2000], height=200)
            context_text = pdf_text

            # Summarize text
            st.subheader("Summarize the PDF Content")
            if st.button("Summarize PDF", use_container_width=True):
                with st.spinner("Summarizing text..."):
                    summary = summarize_text(pdf_text)
                st.success("Summary generated!")
                st.write(summary)
                st.session_state.history.append(("PDF Upload", summary))
        else:
            st.error("Failed to extract text. Please check your PDF file.")

elif option == "Enter Text Manually":
    manual_text = st.text_area("Enter your text below:", height=200)
    if manual_text.strip():
        context_text = manual_text

        st.subheader("Summarize the Entered Text")
        if st.button("Summarize Text", use_container_width=True):
            with st.spinner("Summarizing text..."):
                summary = summarize_text(manual_text)
            st.success("Summary generated!")
            st.write(summary)
            st.session_state.history.append(("Manual Text", summary))
    else:
        st.info("Please enter some text to summarize.")

elif option == "Upload Audio":
    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac"], label_visibility="collapsed")

    if audio_file:
        with st.spinner("Transcribing audio to text..."):
            try:
                transcription = audio_to_text(audio_file)
                st.success("Transcription successful!")
                st.write(transcription)
                st.session_state.history.append(("Audio Upload", transcription))
            except Exception as e:
                st.error(f"Error: {e}")

elif option == "Upload Image":
    image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if image_file:
        with st.spinner("Extracting text from image..."):
            image_text = image_to_text(image_file)
            st.success("Text extracted from image!")
            st.write(image_text)
            st.session_state.history.append(("Image Upload", image_text))

# Sidebar for Interaction History with improved layout
st.sidebar.subheader("Interaction History")
if st.session_state.history:
    for i, (user_input, response_output) in enumerate(st.session_state.history):
        st.sidebar.write(f"**Interaction {i + 1}:**")
        st.sidebar.write(f"**User Input:** {user_input}")
        st.sidebar.write(f"**Response Output:** {response_output}")
else:
    st.sidebar.write("No history yet.")

# Translation Section with clean layout
st.subheader("Translate Text")

# Choose translation direction (English ↔ Chinese)
target_language = st.selectbox("Choose translation direction:", ("English to Chinese", "Chinese to English"))

if context_text:
    st.subheader("Translate the Text")
    if st.button("Translate Text", use_container_width=True):
        with st.spinner("Translating text..."):
            translation_result = translation_model.predict(inputs={"text": context_text})
            translated_text = translation_result.get("translation_text", "")

        st.success(f"Translated text ({target_language}):")
        st.write(translated_text)
        st.session_state.history.append(("Translation", translated_text))

# Add a Conversation AI section with Llama 2 model
st.subheader("Chat with Botify")

# User input for chat
user_query = st.text_input("Enter your query:", key="chat_input", placeholder="Type something to chat!")

# Process the query if entered
if user_query:
    with st.spinner("Generating response..."):
        response_result = llama_model.predict(inputs={"text": user_query})
        bot_response = response_result.get("response_text", "")

    # Display the response
    st.markdown(f"**Botify:** {bot_response}")
    st.session_state.history.append(("User Query", bot_response))
