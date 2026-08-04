"""
src/classical_ml/decision_tree_model.py

Phase 4 — Machine Learning classique — bloc "Decision Tree".

Théorie résumée :
Un arbre de décision découpe récursivement l'espace des features en
posant des questions binaires successives ("le score TF-IDF du mot
'boring' dépasse-t-il 0.15 ?"). Chaque nœud choisit la question qui
sépare le mieux les classes selon un critère d'impureté (Gini par
défaut, ou entropie). Les feuilles finales portent la prédiction.

Force : totalement interprétable -- on peut littéralement lire le
chemin de décision menant à une prédiction.
Faiblesse : très instable (un petit changement de données change tout
l'arbre) et sujet au surapprentissage s'il n'est pas limité en
profondeur.

Vérifié empiriquement sur movie_reviews (TF-IDF 2000 features) :
accuracy = 0.682 -- le PLUS FAIBLE de tous les modèles de la Phase 4.
C'est attendu : un arbre unique sur des données très haute dimension
et creuses ne peut poser qu'un nombre limité de questions pertinentes.
Random Forest (random_forest_model.py) corrige précisément ce défaut.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier


def train_decision_tree(
    X_train: list[str],
    y_train: list[str],
    max_features: int = 2000,
    max_depth: int | None = None,
    random_state: int = 42,
):
    """Vectorise en TF-IDF puis entraîne un arbre de décision.
    max_features réduit à 2000 par défaut (vs 5000 pour les modèles
    linéaires) : les modèles à base d'arbres sont bien plus lents sur
    des données très haute dimension.
    max_depth=None laisse l'arbre croître sans limite -- source
    principale de surapprentissage, à contraindre si besoin."""
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    model.fit(X_train_vec, y_train)
    return model, vectorizer


def evaluate_model(model, vectorizer, X_test: list[str], y_test: list[str]) -> dict:
    """Évalue un modèle déjà entraîné sur un jeu de test."""
    X_test_vec = vectorizer.transform(X_test)
    predictions = model.predict(X_test_vec)
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "report": classification_report(y_test, predictions, output_dict=True),
        "predictions": predictions,
    }
