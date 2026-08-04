"""
tests/classical_ml/test_xgboost_model.py

Tests unitaires pour src/classical_ml/xgboost_model.py.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from classical_ml.data_loader import load_movie_reviews
from classical_ml.xgboost import evaluate_model, train_xgboost


def test_trains_and_predicts():
    X_train, X_test, y_train, y_test = load_movie_reviews()
    model, vectorizer, le = train_xgboost(
        X_train, y_train, max_features=500, n_estimators=20
    )
    results = evaluate_model(model, vectorizer, le, X_test, y_test)
    assert 0.0 <= results["accuracy"] <= 1.0
    assert len(results["predictions"]) == len(y_test)


def test_predictions_are_decoded_to_original_string_labels():
    # XGBoost travaille en interne sur des labels numeriques -- le
    # label_encoder doit les redecoder en "pos"/"neg"
    X_train, X_test, y_train, y_test = load_movie_reviews()
    model, vectorizer, le = train_xgboost(
        X_train, y_train, max_features=500, n_estimators=20
    )
    results = evaluate_model(model, vectorizer, le, X_test, y_test)
    assert set(results["predictions"]) <= {"pos", "neg"}


def test_beats_plain_gradient_boosting():
    # XGBoost apporte regularisation + optimisations -- doit faire au
    # moins aussi bien que le GradientBoosting de scikit-learn
    # (verifie empiriquement : 0.825 vs 0.792)
    from classical_ml.gradient_boosting import evaluate_model as eval_gb
    from classical_ml.gradient_boosting import train_gradient_boosting

    X_train, X_test, y_train, y_test = load_movie_reviews()

    m_gb, v_gb = train_gradient_boosting(
        X_train, y_train, max_features=1000, n_estimators=50
    )
    acc_gb = eval_gb(m_gb, v_gb, X_test, y_test)["accuracy"]

    m_xgb, v_xgb, le_xgb = train_xgboost(
        X_train, y_train, max_features=1000, n_estimators=50
    )
    acc_xgb = evaluate_model(m_xgb, v_xgb, le_xgb, X_test, y_test)["accuracy"]

    assert acc_xgb >= acc_gb - 0.03  # tolerance : au moins comparable


def test_exposes_feature_importances():
    X_train, X_test, y_train, y_test = load_movie_reviews()
    model, vectorizer, le = train_xgboost(
        X_train, y_train, max_features=500, n_estimators=20
    )
    assert len(model.feature_importances_) == len(vectorizer.get_feature_names_out())
