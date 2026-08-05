"""
tests/classical_ml/test_shap_explainer.py

Tests unitaires pour src/classical_ml/shap_explainer.py.
Le test le plus important verifie la propriete d'ADDITIVITE, garantie
mathematique qui distingue SHAP d'heuristiques approximatives.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from classical_ml.data_loader import load_movie_reviews
from classical_ml.logistic_regression import train_logistic_regression
from classical_ml.shap_explainer import (
    build_linear_explainer,
    explain_prediction,
    format_explanation,
    verify_additivity,
)


def _setup():
    X_train, X_test, y_train, y_test = load_movie_reviews()
    model, vectorizer = train_logistic_regression(X_train, y_train, max_features=1000)
    X_train_vec = vectorizer.transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return model, vectorizer, X_train_vec, X_test_vec, y_test


def test_explanation_returns_contributions():
    model, vectorizer, X_train_vec, X_test_vec, y_test = _setup()
    explainer = build_linear_explainer(model, X_train_vec)
    explanation = explain_prediction(
        explainer, X_test_vec, vectorizer.get_feature_names_out(), index=0, top_n=5
    )
    assert len(explanation["contributions"]) <= 5
    assert all(
        "word" in c and "contribution" in c for c in explanation["contributions"]
    )


def test_additivity_property_holds():
    # LA garantie mathematique de SHAP : base + somme = score reel
    model, vectorizer, X_train_vec, X_test_vec, y_test = _setup()
    explainer = build_linear_explainer(model, X_train_vec)
    explanation = explain_prediction(
        explainer, X_test_vec, vectorizer.get_feature_names_out(), index=0
    )
    score_reel = model.decision_function(X_test_vec[0])[0]
    assert verify_additivity(explanation, score_reel, tolerance=1e-3)


def test_contributions_are_sorted_by_absolute_importance():
    model, vectorizer, X_train_vec, X_test_vec, y_test = _setup()
    explainer = build_linear_explainer(model, X_train_vec)
    explanation = explain_prediction(
        explainer, X_test_vec, vectorizer.get_feature_names_out(), index=0, top_n=10
    )
    valeurs_abs = [abs(c["contribution"]) for c in explanation["contributions"]]
    assert valeurs_abs == sorted(valeurs_abs, reverse=True)


def test_pushes_toward_matches_sign():
    model, vectorizer, X_train_vec, X_test_vec, y_test = _setup()
    explainer = build_linear_explainer(model, X_train_vec)
    explanation = explain_prediction(
        explainer, X_test_vec, vectorizer.get_feature_names_out(), index=0
    )
    for c in explanation["contributions"]:
        attendu = "positive" if c["contribution"] > 0 else "negative"
        assert c["pushes_toward"] == attendu


def test_different_reviews_get_different_explanations():
    # l'interet de SHAP : l'explication est SPECIFIQUE a chaque avis,
    # contrairement aux coefficients globaux du modele
    model, vectorizer, X_train_vec, X_test_vec, y_test = _setup()
    explainer = build_linear_explainer(model, X_train_vec)
    noms = vectorizer.get_feature_names_out()

    exp_0 = explain_prediction(explainer, X_test_vec, noms, index=0, top_n=5)
    exp_1 = explain_prediction(explainer, X_test_vec, noms, index=1, top_n=5)

    mots_0 = {c["word"] for c in exp_0["contributions"]}
    mots_1 = {c["word"] for c in exp_1["contributions"]}
    assert mots_0 != mots_1


def test_format_explanation_is_readable():
    model, vectorizer, X_train_vec, X_test_vec, y_test = _setup()
    explainer = build_linear_explainer(model, X_train_vec)
    explanation = explain_prediction(
        explainer, X_test_vec, vectorizer.get_feature_names_out(), index=0, top_n=3
    )
    texte = format_explanation(explanation)
    assert "Valeur de base" in texte
    assert "Score reconstruit" in texte


def test_tree_explainer_handles_three_dimensional_output():
    # regression : TreeExplainer retourne (nb_avis, nb_features, nb_classes)
    # -- trois dimensions -- alors que LinearExplainer en retourne deux.
    # Sans gestion explicite, explain_prediction plantait avec
    # "only 0-dimensional arrays can be converted to Python scalars".
    from classical_ml.random_forest import train_random_forest
    from classical_ml.shap_explainer import build_tree_explainer

    X_train, X_test, y_train, y_test = load_movie_reviews()
    model, vectorizer = train_random_forest(
        X_train, y_train, max_features=300, n_estimators=20
    )
    X_test_vec = vectorizer.transform(X_test).toarray()

    explainer = build_tree_explainer(model)
    explanation = explain_prediction(
        explainer, X_test_vec[:3], vectorizer.get_feature_names_out(), index=0, top_n=5
    )
    assert len(explanation["contributions"]) > 0
    assert all(
        isinstance(c["contribution"], float) for c in explanation["contributions"]
    )
