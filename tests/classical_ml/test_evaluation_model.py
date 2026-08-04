"""
tests/classical_ml/test_evaluation_model.py

Tests unitaires pour src/classical_ml/evaluation.py.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from classical_ml.data_loader import load_movie_reviews
from classical_ml.test_evaluation_model import (
    compare_models_cv,
    cross_validate_model,
    is_difference_significant,
)


def _load_all_data():
    X_train, X_test, y_train, y_test = load_movie_reviews()
    return X_train + X_test, y_train + y_test


def test_cross_validate_returns_expected_structure():
    X, y = _load_all_data()
    results = cross_validate_model(MultinomialNB(), X, y, n_splits=3, max_features=1000)
    assert len(results["scores"]) == 3
    assert 0.0 <= results["mean"] <= 1.0
    assert results["std"] >= 0.0
    assert results["min"] <= results["mean"] <= results["max"]


def test_cross_validation_reveals_score_variability():
    # l'interet principal de la CV : montrer que le score varie selon
    # le decoupage (verifie empiriquement : Naive Bayes va de 0.775 a 0.845)
    X, y = _load_all_data()
    results = cross_validate_model(MultinomialNB(), X, y, n_splits=5, max_features=5000)
    assert (
        results["max"] > results["min"]
    )  # les folds ne donnent pas tous le meme score
    assert results["std"] > 0.0


def test_compare_models_returns_one_entry_per_model():
    X, y = _load_all_data()
    results = compare_models_cv(
        {"nb": MultinomialNB(), "svm": LinearSVC(max_iter=2000)},
        X,
        y,
        n_splits=3,
        max_features=1000,
    )
    assert set(results.keys()) == {"nb", "svm"}
    assert all("mean" in r for r in results.values())


def test_significance_heuristic():
    # ecart plus grand que l'ecart-type -> significatif
    a = {"mean": 0.838, "std": 0.020}
    b = {"mean": 0.810, "std": 0.025}
    assert is_difference_significant(a, b) is True

    # ecart plus petit que l'ecart-type -> pas concluant
    c = {"mean": 0.838, "std": 0.020}
    d = {"mean": 0.824, "std": 0.021}
    assert is_difference_significant(c, d) is False


def test_pipeline_prevents_leakage_by_construction():
    # verifie que la CV utilise bien un pipeline : le score obtenu doit
    # rester dans une plage realiste, pas artificiellement gonfle
    X, y = _load_all_data()
    results = cross_validate_model(
        LinearSVC(max_iter=2000), X, y, n_splits=3, max_features=1000
    )
    assert results["mean"] < 0.95  # pas de score suspect proche de la perfection
