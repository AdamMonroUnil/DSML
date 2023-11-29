import pandas as pd
import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
import joblib

# Load the trained model, tokenizer, and LabelEncoder
model_path = "C:/Users/Adam/OneDrive/Desktop/Uni/UNIL/Semestre 1/Data Science & Machine Learning/my_xlm_roberta_model"
model = XLMRobertaForSequenceClassification.from_pretrained(model_path)
tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
label_encoder = joblib.load(model_path + '/label_encoder.pkl')

# Load your test data
test_data = pd.read_csv('https://raw.githubusercontent.com/AdamMonroUnil/DSML/main/unlabelled_test_data.csv')  # Replace with the path to your test data

# Preprocess test data
def preprocess_function(texts):
    return tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")

test_encodings = preprocess_function(test_data['sentence'].tolist())

# Predicting
model.eval()
with torch.no_grad():
    predictions = model(**test_encodings)
    predicted_labels = torch.argmax(predictions.logits, axis=1)

# Decode the predictions to original labels
decoded_labels = label_encoder.inverse_transform(predicted_labels.numpy())

# Create output DataFrame
output_df = pd.DataFrame({
    'id': test_data['id'],
    'difficulty': decoded_labels
})

# Save to CSV
output_file_path = "C:/Users/Adam/OneDrive/Desktop/Uni/UNIL/Semestre 1/Data Science & Machine Learning/Assignments/DSML/predicted_difficulties.csv"
output_df.to_csv(output_file_path, index=False)
