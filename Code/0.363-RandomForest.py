import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# Load your dataset
df = pd.read_csv('https://raw.githubusercontent.com/AdamMonroUnil/DSML/main/training_data.csv')

# Split the DataFrame into features (X) and target (y)
X = df['sentence']
y = df['difficulty']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a TF-IDF Vectorizer
text_transformer = TfidfVectorizer(ngram_range=(1, 2))

# Using Random Forest Classifier
model = RandomForestClassifier()

# Create a pipeline
pipeline = make_pipeline(text_transformer, model)

# GridSearchCV settings for hyperparameter tuning
param_grid = {
    'randomforestclassifier__n_estimators': [100, 200],
    'randomforestclassifier__max_depth': [None, 10, 20]
}

# Using GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='precision_weighted')

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Predictions and metrics on validation set
y_pred = grid_search.predict(X_val)

# Calculate metrics
print(f"Accuracy: {accuracy_score(y_val, y_pred):.3f}")
print(f"Precision: {precision_score(y_val, y_pred, average='weighted'):.3f}")
print(f"Recall: {recall_score(y_val, y_pred, average='weighted'):.3f}")
print(f"F1-Score: {f1_score(y_val, y_pred, average='weighted'):.3f}")

# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred))


# Load new test data
to_predict = pd.read_csv('https://raw.githubusercontent.com/AdamMonroUnil/DSML/main/unlabelled_test_data.csv')

# Predict using the trained model
predicted_difficulties = grid_search.predict(to_predict['sentence'])

# Create a new DataFrame for submission
submission2 = pd.DataFrame({
    'id': to_predict['id'],
    'difficulty': predicted_difficulties
})

# Save the submission DataFrame to a new CSV file
submission2.to_csv('submission2.csv', index=False)



