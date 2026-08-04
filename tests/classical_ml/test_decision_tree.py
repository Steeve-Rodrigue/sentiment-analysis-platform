"""
tests/classical_ml/test_decision_tree_model.py

Tests unitaires pour src/classical_ml/decision_tree_model.py.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from classical_ml.data_loader import load_movie_reviews
from classical_ml.decision_tree import evaluate_model, train_decision_tree


def test_trains_and_predicts():
    X_train, X_test, y_train, y_test = load_movie_reviews()
    model, vectorizer = train_decision_tree(X_train, y_train, max_features=1000)
    results = evaluate_model(model, vectorizer, X_test, y_test)
    assert 0.0 <= results["accuracy"] <= 1.0
    assert len(results["predictions"]) == len(y_test)


def test_unconstrained_tree_overfits_train():
    # sans max_depth, l'arbre croit jusqu'a memoriser le train --
    # signature du surapprentissage propre a l'arbre unique
    X_train, X_test, y_train, y_test = load_movie_reviews()
    model, vectorizer = train_decision_tree(
        X_train, y_train, max_features=1000, max_depth=None
    )

    from sklearn.metrics import accuracy_score

    X_train_vec = vectorizer.transform(X_train)
    acc_train = accuracy_score(y_train, model.predict(X_train_vec))
    assert acc_train > 0.95


def test_limiting_depth_reduces_train_overfitting():
    X_train, X_test, y_train, y_test = load_movie_reviews()
    from sklearn.metrics import accuracy_score

    m_libre, v_libre = train_decision_tree(
        X_train, y_train, max_features=1000, max_depth=None
    )
    acc_train_libre = accuracy_score(
        y_train, m_libre.predict(v_libre.transform(X_train))
    )

    m_limite, v_limite = train_decision_tree(
        X_train, y_train, max_features=1000, max_depth=5
    )
    acc_train_limite = accuracy_score(
        y_train, m_limite.predict(v_limite.transform(X_train))
    )

    assert acc_train_limite < acc_train_libre
