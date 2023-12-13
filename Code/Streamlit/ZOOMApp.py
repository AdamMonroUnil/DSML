import streamlit as st
import torch
from transformers import CamembertTokenizer, CamembertForSequenceClassification
import joblib
import gdown
import os
from googleapiclient.discovery import build
from langdetect import detect
import requests
from xml.etree import ElementTree

# Function to download a file from Google Drive
def download_file_from_google_drive(url, output_path):
    if not os.path.exists(output_path):
        gdown.download(url, output_path, quiet=False)

# URLs for the model and label encoder on Google Drive
model_url = 'https://drive.google.com/uc?id=1Yy6arimJSZt9aY4sFPh4wi1f5-x6UFyO'
label_encoder_url = 'https://drive.google.com/uc?id=1jnJirofV8QkuujkFlwQ6zz2gH06rnvA9'

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

# Setup YouTube API
youtube_api_key = 'API'  
youtube = build('youtube', 'v3', developerKey=youtube_api_key)

def get_video_id(video_url):
    return video_url.split('v=')[-1]

def get_captions_from_url(video_id):
    request = youtube.captions().list(
        part="snippet",
        videoId=video_id
    )
    response = request.execute()

    for item in response.get("items", []):
        if item["snippet"]["language"] == "fr":
            caption_id = item["id"]
            break
    else:
        return None

    caption_request = youtube.captions().download(
        id=caption_id,
        tfmt='ttml'
    )
    caption_response = caption_request.execute()

    # Parsing and extracting text from TTML (XML-based format)
    captions = ''
    root = ElementTree.fromstring(caption_response)
    for elem in root.iter('{http://www.w3.org/ns/ttml}body'):
        for p in elem:
            captions += p.text + ' '

    return captions

# Streamlit interface
import streamlit as st
# ... [Other imports remain the same]

# ... [Your existing function definitions]

# Streamlit interface
def main():
    set_global_background_color("white")

    left_column, right_column = st.columns([1, 2])

    with left_column:
        st.image("https://upload.wikimedia.org/wikipedia/commons/2/2b/Logo_Universit%C3%A9_de_Lausanne.svg", 
                 caption="University of Lausanne")
        st.image("https://1000logos.net/wp-content/uploads/2021/06/Zoom-Logo.png", 
                 caption="Zoom logo")

    with right_column:
        st.title('Welcome to LingoRank!')

        st.write("""
        ### Overview
        LingoRank is a revolutionary startup aimed at enhancing foreign language learning. Our tool helps English speakers to improve their French by reading texts that match their language proficiency level. 

        Finding texts with the appropriate difficulty level (A1 to C2) can be challenging. LingoRank solves this by predicting the difficulty of French texts, aiding learners in choosing materials that are neither too easy nor too hard. 

        Simply enter a French text below, or a YouTube URL for captions, and LingoRank will evaluate its difficulty level, helping you to choose materials that align with your current understanding and learning goals.
        """)

        # Text input for prediction
        user_input = st.text_area("Enter the French text here:", height=200)
        if st.button('Predict Difficulty of Text'):
            if user_input:
                difficulty = predict_difficulty(user_input)
                st.success(f"The predicted difficulty level of the text is: {difficulty}")
            else:
                st.error("Please enter a French text for analysis.")

        # YouTube URL input for caption analysis
        youtube_url = st.text_input("Or enter a YouTube video URL:")
        if st.button('Analyze YouTube Captions'):
            if youtube_url:
                video_id = get_video_id(youtube_url)
                captions = get_captions_from_url(video_id)
                if captions:
                    if detect(captions) == 'fr':
                        difficulty = predict_difficulty(captions)
                        st.success(f"The predicted difficulty level of the captions is: {difficulty}")
                    else:
                        st.error("Captions are not in French.")
                else:
                    st.error("French captions not available for this video or unable to fetch captions.")
            else:
                st.error("Please enter a YouTube URL for analysis.")

if __name__ == "__main__":
    main()
