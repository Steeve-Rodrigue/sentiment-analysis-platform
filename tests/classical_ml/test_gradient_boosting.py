"""
tests/classical_ml/test_gradient_boosting_model.py

Tests unitaires pour src/classical_ml/gradient_boosting_model.py.
Note : n_estimators reduit dans les tests pour limiter la duree --
le boosting ne se parallelise pas (arbres sequentiels).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from classical_ml.data_loader import load_movie_reviews
from classical_ml.gradient_boosting import evaluate_model, train_gradient_boosting


def test_trains_and_predicts():
    X_train, X_test, y_train, y_test = load_movie_reviews()
    model, vectorizer = train_gradient_boosting(
        X_train, y_train, max_features=500, n_estimators=20
    )
    results = evaluate_model(model, vectorizer, X_test, y_test)
    assert 0.0 <= results["accuracy"] <= 1.0
    assert len(results["predictions"]) == len(y_test)


def test_more_estimators_improves_train_fit():
    # propriete du boosting : chaque arbre supplementaire colle davantage
    # aux donnees d'entrainement (contrairement au bagging qui plafonne)
    from sklearn.metrics import accuracy_score

    X_train, X_test, y_train, y_test = load_movie_reviews()

    m_peu, v_peu = train_gradient_boosting(
        X_train, y_train, max_features=500, n_estimators=10
    )
    acc_train_peu = accuracy_score(y_train, m_peu.predict(v_peu.transform(X_train)))

    m_beaucoup, v_beaucoup = train_gradient_boosting(
        X_train, y_train, max_features=500, n_estimators=100
    )
    acc_train_beaucoup = accuracy_score(
        y_train, m_beaucoup.predict(v_beaucoup.transform(X_train))
    )

    assert acc_train_beaucoup > acc_train_peu


def test_learning_rate_affects_fit_speed():
    # learning_rate plus grand = corrections plus agressives = colle
    # plus vite au train pour un meme nombre d'arbres
    from sklearn.metrics import accuracy_score

    X_train, X_test, y_train, y_test = load_movie_reviews()

    m_lent, v_lent = train_gradient_boosting(
        X_train, y_train, max_features=500, n_estimators=20, learning_rate=0.01
    )
    acc_lent = accuracy_score(y_train, m_lent.predict(v_lent.transform(X_train)))

    m_rapide, v_rapide = train_gradient_boosting(
        X_train, y_train, max_features=500, n_estimators=20, learning_rate=0.5
    )
    acc_rapide = accuracy_score(y_train, m_rapide.predict(v_rapide.transform(X_train)))

    assert acc_rapide > acc_lent


def test_exposes_feature_importances():
    X_train, X_test, y_train, y_test = load_movie_reviews()
    model, vectorizer = train_gradient_boosting(
        X_train, y_train, max_features=500, n_estimators=20
    )
    assert len(model.feature_importances_) == len(vectorizer.get_feature_names_out())
