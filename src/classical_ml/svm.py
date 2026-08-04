"""
src/classical_ml/svm_model.py

Phase 4 — Machine Learning classique — bloc "Linear SVM".

Contrairement à la régression logistique (qui optimise une vraisemblance
probabiliste), un SVM cherche la frontière qui MAXIMISE LA MARGE --
la distance entre la frontière et les points les plus proches de chaque
classe (les "vecteurs de support"). L'intuition : parmi toutes les
lignes qui séparent correctement deux nuages de points, choisir celle
qui laisse le plus d'espace de chaque côté, donc la plus robuste face à
de nouveaux points proches de la frontière.

Pourquoi le noyau LINÉAIRE ici : avec des données en très haute
dimension (des milliers de mots en colonnes via TF-IDF), les classes
sont souvent déjà linéairement séparables -- des noyaux plus complexes
(RBF, polynomial) seraient plus lents sans gain réel.

Vérifié empiriquement sur movie_reviews : Linear SVM (0.835) fait
légèrement mieux que Logistic Regression (0.828) et Naive Bayes (0.805).

Note d'implémentation : LinearSVC (solveur liblinear, optimisé pour les
données creuses) n'expose PAS le nombre de vecteurs de support,
contrairement à SVC(kernel="linear") (solveur libsvm) -- compromis
délibéré : vitesse sur grand vocabulaire contre introspection.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import LinearSVC


def train_linear_svm(
    X_train: list[str],
    y_train: list[str],
    max_features: int = 5000,
    C: float = 1.0,
    max_iter: int = 2000,
):
    """Vectorise en TF-IDF puis entraîne un SVM linéaire. Le paramètre C
    joue le même rôle de régularisation que pour la régression
    logistique (petit = contraint, grand = libre)."""
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    model = LinearSVC(C=C, max_iter=max_iter)
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
