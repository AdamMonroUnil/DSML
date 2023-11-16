import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Download necessary NLTK data
nltk.download('stopwords')

# Define a function for text preprocessing
def preprocess_text(text):
    # Tokenization, stemming, and stopword removal
    stemmer = PorterStemmer()
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# Load your dataset
df = pd.read_csv('https://raw.githubusercontent.com/AdamMonroUnil/DSML/main/training_data.csv')

# Preprocess the text data
df['processed_sentence'] = df['sentence'].apply(preprocess_text)

# Split the DataFrame into features (X) and target (y)
X = df['processed_sentence']
y = df['difficulty']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a TF-IDF Vectorizer and Logistic Regression model
text_transformer = TfidfVectorizer(ngram_range=(1, 2))
model = LogisticRegression(max_iter=1000)

# Create a pipeline
pipeline = make_pipeline(text_transformer, model)

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = pipeline.predict(X_val)

# Calculate metrics
print(f"Accuracy: {accuracy_score(y_val, y_pred):.3f}")
print(f"Precision: {precision_score(y_val, y_pred, average='weighted'):.3f}")
print(f"Recall: {recall_score(y_val, y_pred, average='weighted'):.3f}")
print(f"F1-Score: {f1_score(y_val, y_pred, average='weighted'):.3f}")

# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred))

# Load new test data and predict using the trained model
to_predict = pd.read_csv('https://raw.githubusercontent.com/AdamMonroUnil/DSML/main/unlabelled_test_data.csv')
to_predict['processed_sentence'] = to_predict['sentence'].apply(preprocess_text)
predicted_difficulties = pipeline.predict(to_predict['processed_sentence'])

# Create a new DataFrame for submission
submission = pd.DataFrame({
    'id': to_predict['id'],
    'difficulty': predicted_difficulties
})

# Save the submission DataFrame to a new CSV file
submission.to_csv('submission1.csv', index=False)
