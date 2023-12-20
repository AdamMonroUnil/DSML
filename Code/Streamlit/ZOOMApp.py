import streamlit as st
import torch
from transformers import CamembertTokenizer, CamembertForSequenceClassification
import joblib
import gdown
import os
from langdetect import detect
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound

# Function to download a file from Google Drive
def download_file_from_google_drive(url, output_path):
    if not os.path.exists(output_path):
        gdown.download(url, output_path, quiet=False)

# URLs for the model and label encoder on Google Drive
model_url = 'https://drive.google.com/file/d/1oNMtLm-KL68iv8dkmzrxa_GWpEkGvDJ1'
label_encoder_url = 'https://drive.google.com/file/d/1bQxLME1zKIe5zwclYo6RdsTtw6Aff_hQ'

# Paths where to save the model and label encoder
model_path = 'camembert_model.pth'
label_encoder_path = 'label_encoder.joblib'

# Download the model and the LabelEncoder
download_file_from_google_drive(model_url, model_path)
download_file_from_google_drive(label_encoder_url, label_encoder_path)

# Load the model and the LabelEncoder using st.cache_resource
@st.cache_resource
def load_model():
    model = CamembertForSequenceClassification.from_pretrained('camembert-base', num_labels=6)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

@st.cache_resource
def load_label_encoder():
    return joblib.load(label_encoder_path)

model = load_model()
label_encoder = load_label_encoder()
tokenizer = CamembertTokenizer.from_pretrained('camembert-base')

def predict_difficulty(text):
    inputs = tokenizer(text, padding='max_length', max_length=128, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
    difficulty = label_encoder.inverse_transform([predictions.item()])[0]
    return difficulty

def analyze_subtitles(subtitles):
    if detect(subtitles) == 'fr':
        return predict_difficulty(subtitles)
    else:
        return "Subtitles are not in French."

def fetch_and_analyze_subtitles(video_url):
    video_id = video_url.split('v=')[-1]  # Extract video ID from URL
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = transcript_list.find_transcript(['fr']).fetch()
        subtitles = ' '.join([entry['text'] for entry in transcript])
        return analyze_subtitles(subtitles)
    except NoTranscriptFound:
        return "No subtitles found for this video."
    except Exception as e:
        return str(e)

# Set the global background color to white
def set_global_background_color(background_color):
    st.markdown(
        f"""
        <style>
        .reportview-container .main .block-container{{
            background-color: {background_color};
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_global_background_color("white")  # Set global background to white

# Creating columns for layout
left_column, right_column = st.columns([1, 2])

# Using the left column for logos with a light blue background
with left_column:
    st.markdown(
        f"""
        <div style="background-color:#F0F8FF;padding:10px;border-radius:5px;">
            <img src="https://upload.wikimedia.org/wikipedia/commons/2/2b/Logo_Universit%C3%A9_de_Lausanne.svg" 
                 alt="University of Lausanne logo" style="width:100%">
            <img src="https://1000logos.net/wp-content/uploads/2021/06/Zoom-Logo.png" 
                 alt="Zoom logo" style="width:100%">
        </div>
        <div style="background-color:#F0F8FF;height:640px;"></div>
        """,
        unsafe_allow_html=True
    )

# Using the right column for the main app interface
with right_column:
    st.title('Welcome to LingoRank!')

    st.write("""
    ### Overview
    LingoRank is a revolutionary startup aimed at enhancing foreign language learning. Our tool helps English speakers to improve their French by reading texts that match their language proficiency level. 

    Finding texts with the appropriate difficulty level (A1 to C2) can be challenging. LingoRank solves this by predicting the difficulty of French texts, aiding learners in choosing materials that are neither too easy nor too hard. 

    Simply enter a French text below, and LingoRank will evaluate its difficulty level, helping you to choose materials that align with your current understanding and learning goals.
    """)

    # Text input for prediction
    user_input = st.text_area("Enter the French text here:", height=80)
    if st.button('Predict Difficulty of Text'):
        if user_input:
            difficulty = predict_difficulty(user_input)
            st.success(f"The predicted difficulty level of the text is: {difficulty}")
        else:
            st.error("Please enter a French text for analysis.")

    #YouTube subtitle extraction and analysis
    st.write("## Analyze YouTube Video Subtitles")
    youtube_url = st.text_input("Enter the YouTube video URL:")
    if st.button('Analyze YouTube Subtitles'):
        if youtube_url:
            result = fetch_and_analyze_subtitles(youtube_url)
            st.success(f"The subtitles were predicted as having a french level: {result}")
        else:
            st.error("Please enter a YouTube video URL.")
