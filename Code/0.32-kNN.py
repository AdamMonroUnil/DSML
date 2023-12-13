import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load your dataset
df = pd.read_csv('https://raw.githubusercontent.com/AdamMonroUnil/DSML/main/Data/training_data.csv')  

# Split the DataFrame into features (X) and target (y)
X = df['sentence']
y = df['difficulty']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a TF-IDF Vectorizer and kNN model
text_transformer = TfidfVectorizer(ngram_range=(1, 2))
model = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors

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

# Load new test data
to_predict = pd.read_csv('https://raw.githubusercontent.com/AdamMonroUnil/DSML/main/Data/unlabelled_test_data.csv')  
# Predict using the trained model
predicted_difficulties = pipeline.predict(to_predict['sentence'])

# Create a new DataFrame for submission
submission = pd.DataFrame({
    'id': to_predict['id'],  # Replace 'ID' with the actual ID column name if different
    'difficulty': predicted_difficulties
})

# Save the submission DataFrame to a new CSV file
submission.to_csv('submission.csv', index=False)
