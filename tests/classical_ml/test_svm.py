"""
tests/classical_ml/test_svm.py

Tests unitaires pour src/classical_ml/svm_model.py — chaque test
correspond à un résultat vérifié manuellement pendant l'apprentissage.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from classical_ml.data_loader import load_movie_reviews
from classical_ml.svm import evaluate_model, train_linear_svm


def test_achieves_reasonable_accuracy():
    X_train, X_test, y_train, y_test = load_movie_reviews()
    model, vectorizer = train_linear_svm(X_train, y_train)
    results = evaluate_model(model, vectorizer, X_test, y_test)
    assert results["accuracy"] > 0.70


def test_returns_balanced_report():
    X_train, X_test, y_train, y_test = load_movie_reviews()
    model, vectorizer = train_linear_svm(X_train, y_train)
    results = evaluate_model(model, vectorizer, X_test, y_test)
    report = results["report"]
    assert "pos" in report
    assert "neg" in report


def test_learns_interpretable_coefficients():
    # LinearSVC apprend un poids par mot, comme LogisticRegression --
    # doit produire un vecteur de coefficients de la taille du vocabulaire
    X_train, X_test, y_train, y_test = load_movie_reviews()
    model, vectorizer = train_linear_svm(X_train, y_train, max_features=3000)
    assert model.coef_.shape[1] == len(vectorizer.get_feature_names_out())
