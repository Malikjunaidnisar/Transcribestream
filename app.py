# Step 2: Import libraries
import whisper
from transformers import pipeline
import streamlit as st

# Step 3: Load models globally to avoid loading them multiple times
whisper_model = whisper.load_model("base")  # You can choose "small", "medium", "large"
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Step 4: Define transcription and summarization functions
def transcribe_audio(file_path):
    result = whisper_model.transcribe(file_path)
    return result['text']

def summarize_text(text):
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Step 5: Create Streamlit Interface
st.title("Audio Transcription and Summarization")
st.write("Upload an audio or video file to transcribe and summarize its content.")

uploaded_file = st.file_uploader("Choose an audio/video file", type=["mp3", "wav", "mp4"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp_file", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.write("Transcribing...")
    transcribed_text = transcribe_audio("temp_file")
    st.write("Transcription complete.")
    st.write("Transcribed Text:")
    st.text(transcribed_text)

    st.write("Summarizing...")
    summary = summarize_text(transcribed_text)
    st.write("Summary complete.")
    st.write("Summary:")
    st.text(summary)
