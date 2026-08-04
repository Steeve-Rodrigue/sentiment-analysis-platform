"""
tests/classical_ml/test_lightgbm_model.py

Tests unitaires pour src/classical_ml/lightgbm_model.py.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from classical_ml.data_loader import load_movie_reviews
from classical_ml.lightgbm import evaluate_model, train_lightgbm


def test_trains_and_predicts():
    X_train, X_test, y_train, y_test = load_movie_reviews()
    model, vectorizer, le = train_lightgbm(
        X_train, y_train, max_features=500, n_estimators=20
    )
    results = evaluate_model(model, vectorizer, le, X_test, y_test)
    assert 0.0 <= results["accuracy"] <= 1.0
    assert len(results["predictions"]) == len(y_test)


def test_predictions_are_decoded_to_original_string_labels():
    X_train, X_test, y_train, y_test = load_movie_reviews()
    model, vectorizer, le = train_lightgbm(
        X_train, y_train, max_features=500, n_estimators=20
    )
    results = evaluate_model(model, vectorizer, le, X_test, y_test)
    assert set(results["predictions"]) <= {"pos", "neg"}


def test_is_faster_than_xgboost():
    # LightGBM ("Light") doit etre plus rapide que XGBoost grace a sa
    # croissance leaf-wise (verifie empiriquement : 3.3s vs 9.4s)
    from classical_ml.xgboost import train_xgboost

    X_train, X_test, y_train, y_test = load_movie_reviews()

    debut = time.time()
    train_lightgbm(X_train, y_train, max_features=2000, n_estimators=100)
    duree_lgb = time.time() - debut

    debut = time.time()
    train_xgboost(X_train, y_train, max_features=2000, n_estimators=100)
    duree_xgb = time.time() - debut

    assert duree_lgb < duree_xgb


def test_more_leaves_increases_train_fit():
    # num_leaves controle la complexite en croissance leaf-wise --
    # plus de feuilles = modele plus complexe = colle plus au train
    from sklearn.metrics import accuracy_score

    X_train, X_test, y_train, y_test = load_movie_reviews()

    m_peu, v_peu, l_peu = train_lightgbm(
        X_train, y_train, max_features=500, num_leaves=5
    )
    acc_peu = accuracy_score(
        y_train, l_peu.inverse_transform(m_peu.predict(v_peu.transform(X_train)))
    )

    m_bcp, v_bcp, l_bcp = train_lightgbm(
        X_train, y_train, max_features=500, num_leaves=63
    )
    acc_bcp = accuracy_score(
        y_train, l_bcp.inverse_transform(m_bcp.predict(v_bcp.transform(X_train)))
    )

    assert acc_bcp >= acc_peu
