import streamlit as st
import torch
from transformers import CamembertTokenizer, CamembertForSequenceClassification
import joblib
import gdown
import os

# Function to download a file from Google Drive
def download_file_from_google_drive(url, output_path):
    if not os.path.exists(output_path):
        gdown.download(url, output_path, quiet=False)

# URLs for the model and label encoder on Google Drive
model_url = 'YOUR_GOOGLE_DRIVE_MODEL_URL'
label_encoder_url = 'YOUR_GOOGLE_DRIVE_LABEL_ENCODER_URL'

# Paths where to save the model and label encoder
model_path = 'camembert_model.pth'
label_encoder_path = 'label_encoder.joblib'

# Download the model and the LabelEncoder
download_file_from_google_drive(model_url, model_path)
download_file_from_google_drive(label_encoder_url, label_encoder_path)

# Load the model and the LabelEncoder
@st.cache(allow_output_mutation=True)
def load_model():
    model = CamembertForSequenceClassification.from_pretrained('camembert-base', num_labels=6)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

@st.cache(allow_output_mutation=True)
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

# Streamlit interface
st.title('Welcome to LingoRank!')

# Displaying the Zoom Logo
zoom_logo_url = "https://1000logos.net/wp-content/uploads/2021/06/Zoom-Logo.png"
st.image(zoom_logo_url, width=100)

st.write("""
### Overview
LingoRank is a revolutionary startup aimed at enhancing foreign language learning. Our tool helps English speakers to improve their French by reading texts that match their language proficiency level. 

Finding texts with the appropriate difficulty level (A1 to C2) can be challenging. LingoRank solves this by predicting the difficulty of French texts, aiding learners in choosing materials that are neither too easy nor too hard. 

Simply enter a French text below, and LingoRank will evaluate its difficulty level, helping you to choose materials that align with your current understanding and learning goals.
""")

# Text input for prediction
user_input = st.text_area("Enter the French text here:", height=200)
if st.button('Predict Difficulty'):
    if user_input:
        difficulty = predict_difficulty(user_input)
        st.success(f"The predicted difficulty level of the text is: {difficulty}")
    else:
        st.error("Please enter a French text for analysis.")
