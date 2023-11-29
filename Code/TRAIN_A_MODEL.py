# Import libraries
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib


# Load data
training_data = pd.read_csv('https://raw.githubusercontent.com/AdamMonroUnil/DSML/main/training_data.csv')

# Encode labels
label_encoder = LabelEncoder()
training_data['difficulty_encoded'] = label_encoder.fit_transform(training_data['difficulty'])

# Prepare data with multilingual tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

def preprocess_function(examples):
    return tokenizer(examples['sentence'], padding='max_length', max_length=128, truncation=True)

tokenized_data = training_data[['sentence', 'difficulty_encoded']].to_dict(orient='list')
tokenized_data = preprocess_function(tokenized_data)

# Create dataset
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

dataset = Dataset(tokenized_data, training_data['difficulty_encoded'].tolist())

# Split dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Load multilingual BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased', num_labels=len(label_encoder.classes_))

# Training parameters
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train the model
trainer.train()

# Save the model and tokenizer
model_path = "C:/Users/Adam/OneDrive/Desktop/Uni/UNIL/Semestre 1/Data Science & Machine Learning/Assignments/my_bert_model"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
label_encoder_path = model_path + '/label_encoder.pkl'
joblib.dump(label_encoder, label_encoder_path)
