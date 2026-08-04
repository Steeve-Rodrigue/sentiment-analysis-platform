"""
src/classical_ml/evaluation.py

Phase 4 — Machine Learning classique — outils d'évaluation transverses.

Ce module n'est PAS un modèle : ce sont des outils réutilisables par
tous les modèles de la phase (Naive Bayes, Logistic Regression, SVM,
arbres, boosting) pour évaluer leurs performances de façon rigoureuse.

--- VALIDATION CROISÉE ---

--- MATRICE DE CONFUSION ---

--- COURBE ROC ET AUC ---

--- RECHERCHE D'HYPERPARAMÈTRES (GridSearchCV) ---
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline


def cross_validate_model(
    classifier,
    X: list[str],
    y: list[str],
    n_splits: int = 5,
    max_features: int = 5000,
    scoring: str = "accuracy",
    random_state: int = 42,
) -> dict:
    """Évalue un classifieur par validation croisée stratifiée.

    `classifier` doit être un estimateur scikit-learn NON entraîné
    (ex. LinearSVC(), MultinomialNB()) -- pas un modèle déjà fit, et
    pas nos fonctions train_* de ce projet (qui vectorisent en interne).

    La vectorisation TF-IDF est intégrée dans un Pipeline pour éviter
    toute fuite de données entre folds (voir en-tête de fichier).

    StratifiedKFold (plutôt que KFold simple) garantit que chaque fold
    conserve la même proportion pos/neg que le dataset complet.
    """
    pipeline = make_pipeline(
        TfidfVectorizer(max_features=max_features, stop_words="english"),
        classifier,
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)

    return {
        "scores": scores,
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "min": float(scores.min()),
        "max": float(scores.max()),
    }


def compare_models_cv(
    classifiers: dict,
    X: list[str],
    y: list[str],
    n_splits: int = 5,
    max_features: int = 5000,
) -> dict:
    """Compare plusieurs classifieurs par validation croisée sur les
    mêmes données et les mêmes folds. `classifiers` est un dict
    {nom: estimateur non entraîné}.

    Retourne {nom: résultats}, chaque entrée contenant moyenne,
    écart-type et scores individuels -- de quoi juger si un écart entre
    deux modèles est significatif ou dans le bruit."""
    return {
        nom: cross_validate_model(
            clf, X, y, n_splits=n_splits, max_features=max_features
        )
        for nom, clf in classifiers.items()
    }


def is_difference_significant(results_a: dict, results_b: dict) -> bool:
    """Heuristique simple : l'écart entre deux modèles est-il plus grand
    que la variabilité observée ? Compare la différence des moyennes au
    plus grand des deux écarts-types.

    Ce n'est PAS un test statistique rigoureux (un test de Student
    apparié serait plus correct), mais une règle de lecture pratique
    pour éviter de sur-interpréter de petits écarts."""
    diff = abs(results_a["mean"] - results_b["mean"])
    max_std = max(results_a["std"], results_b["std"])
    return diff > max_std


def get_confusion_matrix(
    y_true: list[str], y_pred, labels: list[str] | None = None
) -> dict:
    """Retourne la matrice de confusion avec ses composantes nommées,
    plus lisible qu'un tableau brut.

    Les clés true_negatives / false_positives / false_negatives /
    true_positives supposent un problème binaire où labels[0] est la
    classe "négative" et labels[1] la classe "positive"."""
    labels = labels or ["neg", "pos"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "matrix": cm,
        "labels": labels,
        "true_negatives": int(cm[0][0]),
        "false_positives": int(cm[0][1]),
        "false_negatives": int(cm[1][0]),
        "true_positives": int(cm[1][1]),
    }


def format_confusion_matrix(cm_result: dict) -> str:
    """Formate la matrice de confusion en texte lisible, avec les
    en-têtes de lignes/colonnes explicites (l'ordre lignes=vérité,
    colonnes=prédiction est une source classique de confusion)."""
    labels = cm_result["labels"]
    cm = cm_result["matrix"]
    lignes = [f"{'':12} {'Predit ' + labels[0]:>14} {'Predit ' + labels[1]:>14}"]
    for i, label in enumerate(labels):
        lignes.append(f"{'Vrai ' + label:12} {cm[i][0]:>14} {cm[i][1]:>14}")
    return "\n".join(lignes)


def get_roc_auc(
    model, X_test_vec, y_test: list[str], positive_label: str = "pos"
) -> dict:
    """Calcule la courbe ROC et l'AUC.

    Nécessite un modèle exposant predict_proba() (Naive Bayes,
    Logistic Regression, arbres, boosting) OU decision_function()
    (LinearSVC, qui n'a PAS de predict_proba -- son score de décision
    brut suffit pour la ROC, puisque seul l'ORDRE des scores compte,
    pas leur calibration en probabilité)."""
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        idx_pos = classes.index(positive_label)
        scores = model.predict_proba(X_test_vec)[:, idx_pos]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test_vec)
    else:
        raise TypeError("Le modèle n'expose ni predict_proba ni decision_function.")

    y_binaire = [1 if label == positive_label else 0 for label in y_test]
    fpr, tpr, thresholds = roc_curve(y_binaire, scores)
    auc = roc_auc_score(y_binaire, scores)

    return {
        "auc": float(auc),
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "scores": scores,
    }


def find_best_threshold(roc_result: dict) -> dict:
    """Trouve le seuil maximisant l'indice de Youden (tpr - fpr) --
    c'est-à-dire le point de la courbe ROC le plus éloigné de la
    diagonale du hasard, souvent un bon compromis par défaut.

    Attention : le "meilleur" seuil dépend en réalité du coût métier
    de chaque type d'erreur (rater un avis négatif vs en signaler un à
    tort) -- cette fonction donne un point de départ neutre, pas une
    réponse universelle."""
    youden = roc_result["tpr"] - roc_result["fpr"]
    idx = int(np.argmax(youden))
    return {
        "threshold": float(roc_result["thresholds"][idx]),
        "tpr": float(roc_result["tpr"][idx]),
        "fpr": float(roc_result["fpr"][idx]),
        "youden_index": float(youden[idx]),
    }


# Grille par défaut : explore conjointement la VECTORISATION et le
# CLASSIFIEUR. Les clés "tfidf__*" et "clf__*" ciblent les étapes
# nommées du pipeline construit dans grid_search_model().
DEFAULT_PARAM_GRID = {
    "tfidf__max_features": [2000, 5000, 10000],
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "clf__C": [0.1, 1.0, 10.0],
}


def grid_search_model(
    classifier,
    X_train: list[str],
    y_train: list[str],
    param_grid: dict | None = None,
    n_splits: int = 5,
    scoring: str = "accuracy",
    random_state: int = 42,
    n_jobs: int = -1,
):
    """Recherche exhaustive des meilleurs hyperparamètres par validation
    croisée, sur le TRAIN uniquement.

    `classifier` doit être un estimateur non entraîné. La grille par
    défaut suppose un classifieur exposant un paramètre C (LinearSVC,
    LogisticRegression) -- fournir sa propre `param_grid` sinon
    (ex. {"clf__max_depth": [3, 5, 10]} pour un arbre).

    ATTENTION au coût : le nombre d'entraînements est le produit de
    toutes les valeurs de la grille × n_splits. Une grille de 18
    combinaisons avec 5 folds = 90 entraînements.

    Retourne l'objet GridSearchCV entraîné -- il se comporte comme un
    modèle (predict, predict_proba) en utilisant automatiquement les
    meilleurs paramètres trouvés."""
    param_grid = param_grid if param_grid is not None else DEFAULT_PARAM_GRID

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english")),
            ("clf", classifier),
        ]
    )

    search = GridSearchCV(
        pipeline,
        param_grid,
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state),
        scoring=scoring,
        n_jobs=n_jobs,
    )
    search.fit(X_train, y_train)
    return search


def summarize_grid_search(
    search, X_test: list[str], y_test: list[str], top_n: int = 5
) -> dict:
    """Résume les résultats d'une recherche : meilleurs paramètres,
    score CV, score final sur le test, et les top_n combinaisons.

    L'écart entre best_cv_score et test_accuracy est attendu et sain :
    le score CV est optimiste car la recherche a sélectionné la
    combinaison la plus favorable parmi toutes celles testées."""
    test_accuracy = accuracy_score(y_test, search.predict(X_test))

    resultats = search.cv_results_
    ordre = np.argsort(resultats["rank_test_score"])[:top_n]
    top = [
        {
            "params": resultats["params"][i],
            "mean_score": float(resultats["mean_test_score"][i]),
            "std_score": float(resultats["std_test_score"][i]),
        }
        for i in ordre
    ]

    return {
        "best_params": search.best_params_,
        "best_cv_score": float(search.best_score_),
        "test_accuracy": float(test_accuracy),
        "optimism_gap": float(search.best_score_ - test_accuracy),
        "top_combinations": top,
        "n_combinations_tested": len(resultats["params"]),
    }
