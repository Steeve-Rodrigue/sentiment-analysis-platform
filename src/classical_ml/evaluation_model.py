"""
src/classical_ml/evaluation.py

Phase 4 — Machine Learning classique — outils d'évaluation transverses.

Ce module n'est PAS un modèle : ce sont des outils réutilisables par
tous les modèles de la phase (Naive Bayes, Logistic Regression, SVM,
arbres, boosting) pour évaluer leurs performances de façon rigoureuse.

--- VALIDATION CROISÉE ---


--- MATRICE DE CONFUSION ---

Tableau croisant vérité et prédiction. Son intérêt par rapport à
l'accuracy : elle montre si le modèle est BIAISÉ DANS UNE DIRECTION.

Vérifié empiriquement (Logistic Regression sur movie_reviews) :
    41 faux positifs (avis négatifs prédits positifs)
    28 faux négatifs (avis positifs prédits négatifs)
Le modèle se trompe donc plutôt en annonçant "positif" à tort --
asymétrie invisible dans le score d'accuracy global, mais qui explique
pourquoi le rappel de "neg" (0.830) dépassait celui de "pos" (0.785).

--- COURBE ROC ET AUC ---

Un classifieur ne renvoie pas directement une classe : il calcule une
PROBABILITÉ, puis applique un SEUIL (0.5 par défaut) pour trancher. Ce
seuil est un choix arbitraire. La courbe ROC trace, pour TOUS les
seuils possibles, le taux de vrais positifs contre le taux de faux
positifs. L'AUC (aire sous la courbe) résume le tout :
    AUC = 1.0  -> séparation parfaite
    AUC = 0.5  -> équivalent au hasard
    AUC > 0.8  -> généralement considéré comme bon

Vérifié empiriquement : AUC = 0.907 pour Logistic Regression, nettement
supérieur à son accuracy (0.828). Pas de contradiction : l'accuracy
juge au seuil 0.5 uniquement, l'AUC dit qu'il EXISTE des seuils où le
modèle sépare très bien. Interprétation directe : si on tire au hasard
un avis positif et un négatif, le modèle donne une probabilité plus
élevée au positif dans 90.7% des cas.

"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline


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
