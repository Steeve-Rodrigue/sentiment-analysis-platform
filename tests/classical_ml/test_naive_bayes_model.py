"""
tests/classical_ml/test_naive_bayes_model.py

Tests unitaires pour src/classical_ml/naive_bayes_model.py et
dataset_loader.py — utilise le vrai corpus movie_reviews (nécessite
uv run python scripts/setup_nltk.py avec 'movie_reviews' ajouté, ou
nltk.download('movie_reviews') au préalable).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from classical_ml.data_loader import load_movie_reviews
from classical_ml.naive_bayes_model import evaluate_model, train_naive_bayes


def test_dataset_loader_returns_balanced_split():
    X_train, X_test, y_train, y_test = load_movie_reviews()
    assert len(X_train) + len(X_test) == 2000
    # stratifie -> proportions pos/neg similaires dans train et test
    assert (
        abs(y_train.count("pos") / len(y_train) - y_test.count("pos") / len(y_test))
        < 0.05
    )


def test_naive_bayes_achieves_reasonable_accuracy():
    # seuil delibere bas (0.65) -- ce test verifie que le pipeline
    # fonctionne de bout en bout, pas qu'il atteint une performance
    # optimale (le vrai score observe est ~0.80)
    X_train, X_test, y_train, y_test = load_movie_reviews()
    model, vectorizer = train_naive_bayes(X_train, y_train)
    results = evaluate_model(model, vectorizer, X_test, y_test)
    assert results["accuracy"] > 0.65


def test_evaluate_model_returns_balanced_report():
    X_train, X_test, y_train, y_test = load_movie_reviews()
    model, vectorizer = train_naive_bayes(X_train, y_train)
    results = evaluate_model(model, vectorizer, X_test, y_test)
    report = results["report"]
    assert "pos" in report
    assert "neg" in report
    assert results["predictions"].shape[0] == len(y_test)
