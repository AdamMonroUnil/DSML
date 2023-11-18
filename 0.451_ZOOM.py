import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Chargement des données
def load_data():
    training_data = pd.read_csv('https://raw.githubusercontent.com/AdamMonroUnil/DSML/main/training_data.csv')
    test_data = pd.read_csv('https://raw.githubusercontent.com/AdamMonroUnil/DSML/main/unlabelled_test_data.csv')
    return training_data, test_data

# Définition du vectoriseur TF-IDF en tant que variable globale
vectorizer = TfidfVectorizer(max_features=5000)

# Prétraitement des données
def preprocess_data(data, is_training=True):
    if is_training:
        X = vectorizer.fit_transform(data['sentence'])
        y = data['difficulty']
        return X, y
    else:
        X = vectorizer.transform(data['sentence'])
        return X

# Évaluation des modèles
def evaluate_models(models, X_train, y_train, X_val, y_val):
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        results[name] = {
            "Accuracy": accuracy_score(y_val, predictions),
            "Precision": precision_score(y_val, predictions, average='weighted'),
            "Recall": recall_score(y_val, predictions, average='weighted'),
            "F1 Score": f1_score(y_val, predictions, average='weighted')
        }
    return results

# Recherche par grille pour l'optimisation des hyperparamètres
def grid_search_optimization(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_weighted', verbose=1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_, grid_search.best_score_

# Script principal
def main():
    training_data, test_data = load_data()

    # Prétraitement des données d'entraînement et de test
    X_train, y_train = preprocess_data(training_data, is_training=True)
    X_test = preprocess_data(test_data, is_training=False)

    # Division des données d'entraînement pour la validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Modèles à évaluer
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "kNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier()
    }

    # Évaluation des modèles
    results = evaluate_models(models, X_train_split, y_train_split, X_val_split, y_val_split)
    print(pd.DataFrame(results).transpose())

    # Optimisation des hyperparamètres (ajuster en fonction des résultats)
    # ...

    # Entraînement du modèle de Régression Logistique avec les meilleurs hyperparamètres
    logistic_model = LogisticRegression(C=10, penalty='l2', max_iter=1000)
    logistic_model.fit(X_train, y_train)

    # Prédictions sur les données de test
    predicted_difficulty = logistic_model.predict(X_test)

    # Création d'un DataFrame avec les résultats
    results_df = pd.DataFrame({
        "id": test_data['id'],
        "difficulty": predicted_difficulty
    })

    # Enregistrement des résultats dans un fichier CSV
    results_df.to_csv("predicted_difficulties.csv", index=False)

if __name__ == "__main__":
    main()
