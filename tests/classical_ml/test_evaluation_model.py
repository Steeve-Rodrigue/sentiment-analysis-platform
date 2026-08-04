"""
tests/classical_ml/test_evaluation_model.py

Tests unitaires pour src/classical_ml/evaluation_model.py.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from classical_ml.data_loader import load_movie_reviews
from classical_ml.evaluation_model import (
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


def test_confusion_matrix_components_sum_to_total():
    from classical_ml.evaluation_model import get_confusion_matrix

    y_true = ["pos", "pos", "neg", "neg", "pos"]
    y_pred = ["pos", "neg", "neg", "pos", "pos"]
    cm = get_confusion_matrix(y_true, y_pred)

    total = (
        cm["true_negatives"]
        + cm["false_positives"]
        + cm["false_negatives"]
        + cm["true_positives"]
    )
    assert total == len(y_true)
    assert cm["true_positives"] == 2  # 2 "pos" correctement predits
    assert cm["false_negatives"] == 1  # 1 "pos" predit "neg"


def test_confusion_matrix_reveals_directional_bias():
    # sur le vrai modele : verifie qu'on peut detecter l'asymetrie des
    # erreurs (41 faux positifs vs 28 faux negatifs observes)
    from classical_ml.evaluation_model import get_confusion_matrix
    from classical_ml.logistic_regression import train_logistic_regression

    X_train, X_test, y_train, y_test = load_movie_reviews()
    model, vectorizer = train_logistic_regression(X_train, y_train)
    predictions = model.predict(vectorizer.transform(X_test))

    cm = get_confusion_matrix(y_test, predictions)
    # les deux types d'erreur existent et ne sont pas identiques
    assert cm["false_positives"] > 0
    assert cm["false_negatives"] > 0
    assert cm["false_positives"] != cm["false_negatives"]


def test_format_confusion_matrix_is_readable():
    from classical_ml.evaluation_model import (
        format_confusion_matrix,
        get_confusion_matrix,
    )

    cm = get_confusion_matrix(["pos", "neg"], ["pos", "neg"])
    texte = format_confusion_matrix(cm)
    assert "Predit" in texte
    assert "Vrai" in texte


def test_roc_auc_on_model_with_predict_proba():
    from classical_ml.evaluation_model import get_roc_auc
    from classical_ml.logistic_regression import train_logistic_regression

    X_train, X_test, y_train, y_test = load_movie_reviews()
    model, vectorizer = train_logistic_regression(X_train, y_train)
    result = get_roc_auc(model, vectorizer.transform(X_test), y_test)

    assert 0.5 < result["auc"] <= 1.0  # nettement mieux que le hasard
    assert len(result["fpr"]) == len(result["tpr"])


def test_roc_auc_works_on_linearsvc_without_predict_proba():
    # LinearSVC n'a PAS de predict_proba -- la fonction doit basculer
    # automatiquement sur decision_function
    from classical_ml.evaluation_model import get_roc_auc
    from classical_ml.svm import train_linear_svm

    X_train, X_test, y_train, y_test = load_movie_reviews()
    model, vectorizer = train_linear_svm(X_train, y_train)
    assert not hasattr(model, "predict_proba")

    result = get_roc_auc(model, vectorizer.transform(X_test), y_test)
    assert 0.5 < result["auc"] <= 1.0


def test_find_best_threshold_returns_point_on_curve():
    from classical_ml.evaluation_model import find_best_threshold, get_roc_auc
    from classical_ml.logistic_regression import train_logistic_regression

    X_train, X_test, y_train, y_test = load_movie_reviews()
    model, vectorizer = train_logistic_regression(X_train, y_train)
    roc = get_roc_auc(model, vectorizer.transform(X_test), y_test)
    best = find_best_threshold(roc)

    assert 0.0 <= best["tpr"] <= 1.0
    assert 0.0 <= best["fpr"] <= 1.0
    assert best["youden_index"] > 0  # meilleur que la diagonale du hasard


def test_grid_search_finds_parameters_and_predicts():
    from classical_ml.evaluation_model import grid_search_model

    X_train, X_test, y_train, y_test = load_movie_reviews()
    # grille volontairement minuscule pour garder le test rapide
    petite_grille = {
        "tfidf__max_features": [1000, 2000],
        "clf__C": [0.1, 1.0],
    }
    search = grid_search_model(
        LinearSVC(max_iter=2000),
        X_train,
        y_train,
        param_grid=petite_grille,
        n_splits=3,
    )

    assert "clf__C" in search.best_params_
    assert "tfidf__max_features" in search.best_params_
    assert 0.0 <= search.best_score_ <= 1.0
    # l'objet entraine se comporte comme un modele
    predictions = search.predict(X_test)
    assert len(predictions) == len(y_test)


def test_grid_search_summary_reports_optimism_gap():
    from classical_ml.evaluation_model import grid_search_model, summarize_grid_search

    X_train, X_test, y_train, y_test = load_movie_reviews()
    petite_grille = {"clf__C": [0.1, 1.0], "tfidf__max_features": [1000]}
    search = grid_search_model(
        LinearSVC(max_iter=2000),
        X_train,
        y_train,
        param_grid=petite_grille,
        n_splits=3,
    )
    resume = summarize_grid_search(search, X_test, y_test)

    assert resume["n_combinations_tested"] == 2
    assert 0.0 <= resume["test_accuracy"] <= 1.0
    assert len(resume["top_combinations"]) <= 5
    # l'ecart d'optimisme existe (peut etre positif ou negatif selon le tirage)
    assert "optimism_gap" in resume


def test_grid_search_accepts_custom_grid_for_non_c_models():
    # verifie que la fonction n'est pas verrouillee sur les modeles
    # exposant un parametre C -- un arbre utilise max_depth
    from sklearn.tree import DecisionTreeClassifier

    from classical_ml.evaluation_model import grid_search_model

    X_train, X_test, y_train, y_test = load_movie_reviews()
    grille_arbre = {"clf__max_depth": [3, 5], "tfidf__max_features": [500]}
    search = grid_search_model(
        DecisionTreeClassifier(random_state=42),
        X_train,
        y_train,
        param_grid=grille_arbre,
        n_splits=3,
    )
    assert search.best_params_["clf__max_depth"] in [3, 5]
