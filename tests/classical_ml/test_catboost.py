"""
tests/classical_ml/test_catboost_model.py

Tests unitaires pour src/classical_ml/catboost_model.py.

IMPORTANT : tous les tests utilisent des parametres volontairement
reduits (max_features=300, iterations=20) -- avec les valeurs par
defaut, CatBoost prend ~220s sur ce dataset, ce qui rendrait la suite
de tests inutilisable en CI. Le vrai benchmark complet est dans le
fichier d'experimentation, a lancer manuellement.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from classical_ml.catboost import evaluate_model, train_catboost
from classical_ml.data_loader import load_movie_reviews


def test_trains_and_predicts():
    X_train, X_test, y_train, y_test = load_movie_reviews()
    model, vectorizer, le = train_catboost(
        X_train, y_train, max_features=300, iterations=20
    )
    results = evaluate_model(model, vectorizer, le, X_test, y_test)
    assert 0.0 <= results["accuracy"] <= 1.0
    assert len(results["predictions"]) == len(y_test)


def test_predictions_are_decoded_to_original_string_labels():
    X_train, X_test, y_train, y_test = load_movie_reviews()
    model, vectorizer, le = train_catboost(
        X_train, y_train, max_features=300, iterations=20
    )
    results = evaluate_model(model, vectorizer, le, X_test, y_test)
    assert set(results["predictions"]) <= {"pos", "neg"}


def test_more_iterations_improves_train_fit():
    from sklearn.metrics import accuracy_score

    X_train, X_test, y_train, y_test = load_movie_reviews()

    m_peu, v_peu, l_peu = train_catboost(
        X_train, y_train, max_features=300, iterations=10
    )
    preds_peu = l_peu.inverse_transform(
        m_peu.predict(v_peu.transform(X_train)).ravel().astype(int)
    )
    acc_peu = accuracy_score(y_train, preds_peu)

    m_bcp, v_bcp, l_bcp = train_catboost(
        X_train, y_train, max_features=300, iterations=50
    )
    preds_bcp = l_bcp.inverse_transform(
        m_bcp.predict(v_bcp.transform(X_train)).ravel().astype(int)
    )
    acc_bcp = accuracy_score(y_train, preds_bcp)

    assert acc_bcp >= acc_peu
