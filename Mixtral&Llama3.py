import requests
import streamlit as st
import PyPDF2
from datetime import datetime
from gtts import gTTS  # Import gtts for text-to-speech
import os
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image
import json
from io import BytesIO
import openai
import pytz

# Accessing the Sambanova API key from Streamlit secrets
sambanova_api_key = st.secrets["general"]["SAMBANOVA_API_KEY"]

class SambanovaClient:
    def _init_(self, api_key, base_url):
        # Initialize with API key and base URL
        self.api_key = api_key
        self.base_url = base_url

    def chat(self, model, messages, temperature=0.7, top_p=1.0, max_tokens=500):
        # Prepare headers and data for the API request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Request payload
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p
        }

        try:
            # Send POST request to the Sambanova API
            response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=data)
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            result = response.json()
            return result['choices'][0]['message']['content']  # Return the response content
        except requests.exceptions.RequestException as e:
            # Catch errors and return a useful message
            raise Exception(f"Error while calling Sambanova API: {str(e)}")

# Initialize SambanovaClient with the API key fetched from secrets
base_url = "https://api.sambanova.ai/v1"  # Correct Sambanova URL
client = SambanovaClient(api_key=sambanova_api_key, base_url=base_url)

# Hugging Face BLIP-2 Setup
hf_token = "hf_rLRfVDnchDCuuaBFeIKTAbrptaNcsHUNM"
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large", token=hf_token)
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", token=hf_token)

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
    "Authorization": f"Bearer {api_key}",  # Use api_key here, not groqapi_key
    "Content-Type": "application/json"
}

# Available models, including the two new Sambanova models
available_models = {
    "Mixtral 8x7b": "mixtral-8x7b-32768",
    "Llama 3.1 70b Versatile": "llama-3.1-70b-versatile",
    "Qwen 2.5-72B-Instruct": "sambanova-qwen-2.5-72b-instruct",
    "Meta-Llama 3.2-1B-Instruct": "sambanova-meta-llama-3.2-1b-instruct"
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
        "max_tokens": 5000,
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

# Function to Translate Text Using the Selected Model
def translate_text(text, target_language, model_id):
    url = f"{base_url}/chat/completions"
    data = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": f"Translate the following text into {target_language}."},
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
            return f"Translation error: {response.status_code}"
    except requests.exceptions.RequestException as e:
        return f"An error occurred during translation: {e}"

# Updated function to transcribe audio using the Groq Whisper API
def transcribe_audio(file):
    whisper_api_key = st.secrets["whisper"]["WHISPER_API_KEY"]  # Access Whisper API key
    url = "https://api.groq.com/openai/v1/audio/transcriptions"  # Groq transcription endpoint

    # Check file type
    valid_types = ['flac', 'mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'ogg', 'opus', 'wav', 'webm']
    extension = file.name.split('.')[-1].lower()
    if extension not in valid_types:
        st.error(f"Invalid file type: {extension}. Supported types: {', '.join(valid_types)}")
        return None

    # Prepare file buffer with proper extension in the .name attribute
    audio_data = file.read()  # Use file.read() to handle the uploaded file correctly
    buffer = BytesIO(audio_data)
    buffer.name = f"file.{extension}"  # Assigning a valid extension based on the uploaded file

    # Prepare the request payload
    headers = {"Authorization": f"Bearer {whisper_api_key}"}
    data = {"model": "whisper-large-v3-turbo", "language": "en"}

    try:
        # Send the audio file for transcription
        response = requests.post(
            url,
            headers=headers,
            files={"file": buffer},
            data=data
        )

        # Handle response
        if response.status_code == 200:
            transcription = response.json()
            return transcription.get("text", "No transcription text found.")
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error during transcription: {str(e)}")
        return None

# Step 2: Function to Extract Text from Image using BLIP-2
def extract_text_from_image(image_file):
    # Open image from uploaded file
    image = Image.open(image_file)

    # Preprocess the image for the BLIP-2 model
    inputs = blip_processor(images=image, return_tensors="pt")

    # Generate the caption (text) for the image
    out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)

    return caption

# Input Method Selection
input_method = st.selectbox("Select Input Method", ["Upload PDF", "Enter Text Manually", "Upload Audio", "Upload Image"])

# Model selection - Available only for PDF and manual text input
if input_method in ["Upload PDF", "Enter Text Manually"]:
    selected_model_name = st.selectbox("Choose a model:", list(available_models.keys()), key="model_selection")
    
    # Ensure that the user selects a model (no default)
    if selected_model_name:
        selected_model_id = available_models[selected_model_name]
    else:
        st.error("Please select a model to proceed.")
        selected_model_id = None
else:
    selected_model_id = None

# Sidebar for interaction history
if "history" not in st.session_state:
    st.session_state.history = []

# Initialize content variable
content = ""

# Language selection for translation
languages = [
    "English", "Chinese", "Spanish", "French", "Italian", "Portuguese", "Romanian", 
    "German", "Dutch", "Swedish", "Danish", "Norwegian", "Russian", 
    "Polish", "Czech", "Ukrainian", "Serbian", "Japanese", 
    "Korean", "Hindi", "Bengali", "Arabic", "Hebrew", "Persian", 
    "Punjabi", "Tamil", "Telugu", "Swahili", "Amharic"
]
selected_language = st.selectbox("Choose your preferred language for output", languages)

# Handle different input methods
if input_method == "Upload PDF":
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file:
        # Extract text from the uploaded PDF
        st.write("Extracting text from the uploaded PDF...")
        pdf_text = extract_text_from_pdf(uploaded_file)
        st.success("Text extracted successfully!")

        # Display extracted text with adjusted font size
        with st.expander("View Extracted Text"):
            st.markdown(f"<div style='font-size: 14px;'>{pdf_text}</div>", unsafe_allow_html=True)

        # Assign extracted text to content for chat
        content = pdf_text

        # Summarize the extracted text only when the button is clicked
        if st.button("Summarize Text"):
            st.write("Summarizing the text...")
            summary = summarize_text(pdf_text, selected_model_id)
            st.write("Summary:")
            st.write(summary)

            # Translate the summary to the selected language
            translated_summary = translate_text(summary, selected_language, selected_model_id)
            st.write(f"Translated Summary in {selected_language}:")
            st.write(translated_summary)

            # Convert summary to audio in English (not translated)
            tts = gTTS(text=summary, lang='en')  # Use English summary for audio
            tts.save("response.mp3")
            st.audio("response.mp3", format="audio/mp3")

elif input_method == "Enter Text Manually":
    manual_text = st.text_area("Enter your text manually:")

    if manual_text:
        # Assign entered text to content for chat
        content = manual_text

        if st.button("Summarize Text"):
            st.write("Summarizing the entered text...")
            summary = summarize_text(manual_text, selected_model_id)
            st.write("Summary:")
            st.write(summary)

            # Translate the summary to the selected language
            translated_summary = translate_text(summary, selected_language, selected_model_id)
            st.write(f"Translated Summary in {selected_language}:")
            st.write(translated_summary)

            # Convert summary to audio in English (not translated)
            tts = gTTS(text=summary, lang='en')  # Use English summary for audio
            tts.save("response.mp3")
            st.audio("response.mp3", format="audio/mp3")

# Model selection for Image and Audio uploads
if input_method in ["Upload Audio", "Upload Image"]:
    selected_model_name = st.selectbox("Choose a model:", list(available_models.keys()), key="model_selection_image_audio")
    
    # Ensure that the user selects a model (no default)
    if selected_model_name:
        selected_model_id = available_models[selected_model_name]
    else:
        st.error("Please select a model to proceed.")
        selected_model_id = None
else:
    selected_model_id = None

# Handle image and audio inputs with selected model
if input_method == "Upload Image":
    uploaded_image = st.file_uploader("Upload an image file", type=["jpg", "png"])

    if uploaded_image:
        st.write("Image uploaded. Extracting text using BLIP-2...")
        try:
            # Extract text using BLIP-2
            image_text = extract_text_from_image(uploaded_image)
            st.success("Text extracted successfully!")

            # Display extracted text with adjusted font size
            with st.expander("View Extracted Text"):
                st.markdown(f"<div style='font-size: 14px;'>{image_text}</div>", unsafe_allow_html=True)

            content = image_text

            # Summarize the extracted text
            if st.button("Summarize and Translate Text"):
                st.write("Summarizing the extracted text...")
                summary = summarize_text(image_text, selected_model_id)
                st.write("Summary:")
                st.write(summary)

                # Translate the summary to the selected language
                translated_summary = translate_text(summary, selected_language, selected_model_id)
                st.write(f"Translated Summary in {selected_language}:")
                st.write(translated_summary)

                # Convert summary to audio in English (not translated)
                tts = gTTS(text=summary, lang='en')  # Use English summary for audio
                tts.save("response.mp3")
                st.audio("response.mp3", format="audio/mp3")

                # Store the summarized content
                st.session_state.summarized_content = summary

                # Now allow model selection for image processing
                st.session_state.image_model_selected = True

        except Exception as e:
            st.error(f"Error extracting text from image: {e}")

    # Only display model selection after summarization
    if 'summarized_content' in st.session_state and st.session_state.summarized_content:
        selected_model_name = st.selectbox("Choose a model:", list(available_models.keys()), key="model_selection_image_audio")
        
        # Ensure that the user selects a model (no default)
        if selected_model_name:
            selected_model_id = available_models[selected_model_name]
        else:
            st.error("Please select a model to proceed.")
            selected_model_id = None

elif input_method == "Upload Audio":
    uploaded_audio = st.file_uploader("Upload an audio file", type=["mp3", "wav"])

    if uploaded_audio:
        st.write("Audio file uploaded. Processing audio...")

        # Transcribe using Groq's Whisper API
        transcript = transcribe_audio(uploaded_audio)
        if transcript:
            st.write("Transcription:")
            st.write(transcript)

            content = transcript

            # Summarize the transcribed text
            if st.button("Summarize and Translate Text"):
                st.write("Summarizing the transcribed text...")
                summary = summarize_text(transcript, selected_model_id)
                st.write("Summary:")
                st.write(summary)

                # Translate the summary to the selected language
                translated_summary = translate_text(summary, selected_language, selected_model_id)
                st.write(f"Translated Summary in {selected_language}:")
                st.write(translated_summary)

                # Convert summary to audio in English (not translated)
                tts = gTTS(text=summary, lang='en')  # Use English summary for audio
                tts.save("response.mp3")
                st.audio("response.mp3", format="audio/mp3")

                # Store the summarized content
                st.session_state.summarized_content = summary

                # Now allow model selection for audio processing
                st.session_state.audio_model_selected = True

        else:
            st.error("Failed to transcribe the audio.")

    # Only display model selection after summarization
    if 'summarized_content' in st.session_state and st.session_state.summarized_content:
        selected_model_name = st.selectbox("Choose a model:", list(available_models.keys()), key="model_selection_audio")
        
        # Ensure that the user selects a model (no default)
        if selected_model_name:
            selected_model_id = available_models[selected_model_name]
        else:
            st.error("Please select a model to proceed.")
            selected_model_id = None


# Step 2: User Input for Questions
if content:
    question = st.text_input("Ask a question about the content:")

    if question:
        # Create interaction dictionary with timestamp
        # Define Malaysia Time (MYT) timezone
        malaysia_timezone = pytz.timezone("Asia/Kuala_Lumpur")

        # Get the current time in Malaysia Time (MYT)
        current_time = datetime.now(malaysia_timezone).strftime("%Y-%m-%d %H:%M:%S")
        
        interaction = {
            "time": current_time,
            "input_method": input_method,
            "question": question,
            "response": "",
            "content_preview": content[:100] if content else "No content available"
        }
        # Add user question to history
        st.session_state.history.append(interaction)

        if content:
            # Send the question and content to the API for response
            url = f"{base_url}/chat/completions"
            data = {
                "model": selected_model_id,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant. Use the following content to answer the user's questions."},
                    {"role": "system", "content": content},
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
                    answer = result['choices'][0]['message']['content']

                    # Store bot's answer in the interaction history
                    interaction["response"] = answer
                    st.session_state.history[-1] = interaction

                    # Display bot's response
                    st.write("Answer:", answer)

                else:
                    st.write(f"Error {response.status_code}: {response.text}")
            except requests.exceptions.RequestException as e:
                st.write(f"An error occurred: {e}")

# Display interaction history in the sidebar
if st.session_state.history:
    st.sidebar.header("Interaction History")
    for idx, interaction in enumerate(st.session_state.history):
        st.sidebar.markdown(f"{interaction['time']}")
        st.sidebar.markdown(f"*Input Method*: {interaction['input_method']}")
        st.sidebar.markdown(f"*Question*: {interaction['question']}")
        st.sidebar.markdown(f"*Response*: {interaction['response']}")
        st.sidebar.markdown(f"*Content Preview*: {interaction['content_preview']}")
        st.sidebar.markdown("---")
