"""
tests/classical_ml/test_random_forest_model.py

Tests unitaires pour src/classical_ml/random_forest_model.py.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from classical_ml.data_loader import load_movie_reviews
from classical_ml.random_forest import evaluate_model, train_random_forest


def test_trains_and_predicts():
    X_train, X_test, y_train, y_test = load_movie_reviews()
    model, vectorizer = train_random_forest(
        X_train, y_train, max_features=1000, n_estimators=50
    )
    results = evaluate_model(model, vectorizer, X_test, y_test)
    assert 0.0 <= results["accuracy"] <= 1.0
    assert len(results["predictions"]) == len(y_test)


def test_forest_beats_single_tree():
    # verifie le principe du bagging : la foret doit corriger
    # l'instabilite de l'arbre unique (verifie : 0.792 vs 0.682)
    from classical_ml.decision_tree import evaluate_model as eval_tree
    from classical_ml.decision_tree import train_decision_tree

    X_train, X_test, y_train, y_test = load_movie_reviews()

    m_tree, v_tree = train_decision_tree(X_train, y_train, max_features=1000)
    acc_tree = eval_tree(m_tree, v_tree, X_test, y_test)["accuracy"]

    m_forest, v_forest = train_random_forest(
        X_train, y_train, max_features=1000, n_estimators=50
    )
    acc_forest = evaluate_model(m_forest, v_forest, X_test, y_test)["accuracy"]

    assert acc_forest > acc_tree


def test_more_trees_does_not_degrade_performance():
    # propriete du bagging : ajouter des arbres ne fait pas surapprendre
    # (contrairement au boosting), la performance plafonne au pire
    X_train, X_test, y_train, y_test = load_movie_reviews()

    m_10, v_10 = train_random_forest(
        X_train, y_train, max_features=1000, n_estimators=10
    )
    acc_10 = evaluate_model(m_10, v_10, X_test, y_test)["accuracy"]

    m_100, v_100 = train_random_forest(
        X_train, y_train, max_features=1000, n_estimators=100
    )
    acc_100 = evaluate_model(m_100, v_100, X_test, y_test)["accuracy"]

    assert acc_100 >= acc_10 - 0.05  # tolerance : pas de degradation nette


def test_exposes_feature_importances():
    X_train, X_test, y_train, y_test = load_movie_reviews()
    model, vectorizer = train_random_forest(
        X_train, y_train, max_features=1000, n_estimators=50
    )
    assert len(model.feature_importances_) == len(vectorizer.get_feature_names_out())
