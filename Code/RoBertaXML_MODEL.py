import pandas as pd
import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Load data
training_data = pd.read_csv('https://raw.githubusercontent.com/AdamMonroUnil/DSML/main/training_data.csv')

# Confirm that 'sentence' column is present
print("Columns in DataFrame:", training_data.columns)
assert 'sentence' in training_data.columns, "'sentence' column not found in DataFrame"

# Data Cleaning (optional)
# training_data['sentence'] = training_data['sentence'].str.replace('[^a-zA-Zéèàêâôûùïîç]', ' ', regex=True)

# Encode labels
label_encoder = LabelEncoder()
training_data['difficulty_encoded'] = label_encoder.fit_transform(training_data['difficulty'])

# Split dataset into train and test sets
train_data, test_data = train_test_split(training_data, test_size=0.2, random_state=42)

# Initialize tokenizer (XLM-RoBERTa)
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

def tokenize_function(texts):
    return tokenizer(texts, padding='max_length', max_length=128, truncation=True)

# Tokenize data
train_encodings = tokenize_function(train_data['sentence'].tolist())
test_encodings = tokenize_function(test_data['sentence'].tolist())

# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = Dataset(train_encodings, train_data['difficulty_encoded'].tolist())
test_dataset = Dataset(test_encodings, test_data['difficulty_encoded'].tolist())

# Load pre-trained XLM-RoBERTa model and fine-tune
model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=len(label_encoder.classes_))

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy="epoch"
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Train the model
trainer.train()

# Save the model, tokenizer, and label_encoder
model_path = "C:/Users/Adam/OneDrive/Desktop/Uni/UNIL/Semestre 1/Data Science & Machine Learning/my_xlm_roberta_model"
if not os.path.exists(model_path):
    os.makedirs(model_path)

model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
label_encoder_path = os.path.join(model_path, 'label_encoder.pkl')
joblib.dump(label_encoder, label_encoder_path)
