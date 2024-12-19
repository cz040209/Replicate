import requests
import streamlit as st
import PyPDF2
from datetime import datetime
from gtts import gTTS  # Import gtts for text-to-speech
import os
import pytesseract
from PIL import Image
import json

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

# Set up Rev AI API Key (make sure you have this stored in Streamlit secrets)
rev_ai_api_key = "02iieflkd7QpA8_34YnJJQo2s81yjoLo5Fj9QUAXxdTUYvrfFErqT9m1CM6yVuXmMJdnNzSo4HU5XxEanm-Qe-TzGnjEY"

# Function to Convert Audio to Text Using Rev AI API
def transcribe_audio(rev_ai_api_key, audio_file):
    url = "https://api.rev.ai/speechtotext/v1beta/jobs"
    headers = {
        "Authorization": f"Bearer {rev_ai_api_key}",
        "Content-Type": "application/json"
    }

    # Get the file data directly from the uploaded file
    audio_data = audio_file.getvalue()

    # Upload the file to Rev AI (First, we upload it to get the job ID)
    files = {
        'file': ('audio_file', audio_data, audio_file.type),  # File with its MIME type
    }

    try:
        # Step 1: Upload the audio to Rev AI for processing
        response = requests.post(url, headers=headers, files=files)
        if response.status_code == 200:
            job_data = response.json()
            job_id = job_data["id"]
            st.write(f"Audio uploaded successfully! Job ID: {job_id}")

            # Step 2: Check the status of the job until it's done
            status_url = f"https://api.rev.ai/speechtotext/v1beta/jobs/{job_id}"
            while True:
                status_response = requests.get(status_url, headers=headers)
                status_data = status_response.json()

                if status_data['status'] == 'completed':
                    # Once completed, get the transcript URL
                    transcript_url = status_data['results']['transcript_url']
                    transcript_response = requests.get(transcript_url)
                    transcript = transcript_response.text
                    return transcript
                elif status_data['status'] == 'failed':
                    return f"Error: Transcription job failed. {status_data['error']}"
                else:
                    st.write("Processing... Please wait a moment.")

        else:
            return f"Error {response.status_code}: {response.text}"
    except requests.exceptions.RequestException as e:
        return f"An error occurred during transcription: {e}"


# Input Method Selection
input_method = st.selectbox("Select Input Method", ["Upload PDF", "Enter Text Manually", "Upload Audio", "Upload Image"])

# Initialize content variable
content = ""

# Handle different input methods
if input_method == "Upload Audio":
    uploaded_audio = st.file_uploader("Upload an audio file", type=["mp3", "wav"])

    if uploaded_audio:
        st.write("Audio file uploaded. Processing audio...")
        # Use Rev AI for transcription
        transcript = transcribe_audio(rev_ai_api_key, uploaded_audio)
        st.write("Transcription:")
        st.write(transcript)

elif input_method == "Upload Image":
    uploaded_image = st.file_uploader("Upload an image file", type=["jpg", "png"])

    if uploaded_image:
        st.write("Image uploaded. Extracting text using OCR...")
        try:
            image = Image.open(uploaded_image)
            image_text = pytesseract.image_to_string(image)
            st.success("Text extracted successfully!")

            # Display extracted text with adjusted font size
            with st.expander("View Extracted Text"):
                st.markdown(f"<div style='font-size: 14px;'>{image_text}</div>", unsafe_allow_html=True)

            content = image_text
        except Exception as e:
            st.error(f"Error extracting text from image: {e}")

# Step 2: User Input for Questions
if content:
    question = st.text_input("Ask a question about the content:")

    if question:
        # Create interaction dictionary with timestamp
        interaction = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
        st.sidebar.markdown(f"**{interaction['time']}**")
        st.sidebar.markdown(f"**Input Method**: {interaction['input_method']}")
        st.sidebar.markdown(f"**Question**: {interaction['question']}")
        st.sidebar.markdown(f"**Response**: {interaction['response']}")
        st.sidebar.markdown(f"**Content Preview**: {interaction['content_preview']}")
        st.sidebar.markdown("---")
