import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Chargement et préparation des données d'entraînement
training_data_url = 'https://raw.githubusercontent.com/AdamMonroUnil/DSML/main/training_data_dupplicated.csv'
training_data = pd.read_csv(training_data_url)
label_encoder = LabelEncoder()
training_data['difficulty_encoded'] = label_encoder.fit_transform(training_data['difficulty'])

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-french-europeana-cased")

def preprocess_function(examples):
    return tokenizer(examples, padding='max_length', max_length=128, truncation=True)

tokenized_data = preprocess_function(training_data['sentence'].tolist())

# Création du dataset
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

# Division du dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Chargement du modèle
model = AutoModelForSequenceClassification.from_pretrained("dbmdz/bert-base-french-europeana-cased", num_labels=len(label_encoder.classes_))

# Move model to the specified device
model = model.to(device)

# Paramètres d'entraînement
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Entraîneur
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Entraînement du modèle
trainer.train()

# Préparation des données de test
test_data = pd.read_csv('https://raw.githubusercontent.com/AdamMonroUnil/DSML/main/unlabelled_test_data.csv')
encoded_test_data = preprocess_function(test_data['sentence'].tolist())

# Prédiction
model.eval()
with torch.no_grad():
    inputs = {key: torch.tensor(val).to(device) for key, val in encoded_test_data.items()}  # Move input data to GPU
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)

# Move predictions back to CPU for further processing if needed
predictions = predictions.cpu()

# Convertir les prédictions en étiquettes lisibles
predicted_labels = label_encoder.inverse_transform(predictions.numpy())

# Enregistrement des résultats
results = pd.DataFrame({
    'id': test_data['id'],
    'difficulty': predicted_labels
})
results.to_csv('submission9.csv', index=False)
