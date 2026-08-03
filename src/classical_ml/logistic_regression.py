"""
src/classical_ml/logistic_regression.py

Phase 4 — Machine Learning classique — bloc "Logistic Regression".

La régression logistique applique une somme pondérée des features
(comme une régression linéaire classique), puis l'écrase entre 0 et 1
via la fonction sigmoïde pour obtenir une probabilité :
    P(y=1 | x) = 1 / (1 + exp(-(w.x + b)))
Contrairement à Naive Bayes (indépendance supposée entre mots, poids
calculés séparément), la régression logistique APPREND directement les
poids w pour bien séparer les classes -- pas d'hypothèse simplificatrice.

Vérifié empiriquement sur movie_reviews : Logistic Regression (0.828)
bat légèrement Naive Bayes (0.805) sur le même dataset.

Le paramètre C contrôle la régularisation : C petit = modèle contraint
(poids modestes, moins de surapprentissage) ; C grand = modèle libre
(risque de surapprentissage). Vérifié empiriquement : à C=100,
l'accuracy train atteint 1.000 (mémorisation) mais l'accuracy test
n'est pas meilleure qu'à C=10 -- signe clair de surapprentissage au-delà
d'un certain point.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def train_logistic_regression(
    X_train: list[str], y_train: list[str], max_features: int = 5000, C: float = 1.0
):
    """Vectorise en TF-IDF puis entraîne une régression logistique.
    Retourne le modèle ET le vectorizer (même raison que pour Naive
    Bayes : nécessaire pour transformer les données de test de façon
    cohérente, jamais re-fit sur de nouvelles données)."""
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000, C=C)
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
