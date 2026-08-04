"""
src/classical_ml/evaluation.py

Phase 4 — Machine Learning classique — outils d'évaluation transverses.

Ce module n'est PAS un modèle : ce sont des outils réutilisables par
tous les modèles de la phase (Naive Bayes, Logistic Regression, SVM,
arbres, boosting) pour évaluer leurs performances de façon rigoureuse.

--- VALIDATION CROISÉE ---

Théorie résumée (voir la conversation associée pour le détail) :
Un score obtenu sur UN SEUL découpage train/test est fragile -- il
dépend du hasard du tirage. La K-fold cross-validation découpe les
données en K parts, entraîne K fois (chaque part servant une fois de
test), et retourne la moyenne ET l'écart-type des K scores.

Vérifié empiriquement sur movie_reviews (5 folds) :
    Naive Bayes          0.810 ± 0.025  (scores : 0.775 à 0.845 !)
    Logistic Regression  0.824 ± 0.021
    Linear SVM           0.838 ± 0.020

L'enseignement clé : Naive Bayes varie de 7 POINTS selon le découpage.
L'écart SVM vs LogisticRegression (1.4 pt) est plus PETIT que
l'écart-type (~2 pts) -- cette différence n'est donc pas clairement
significative. En revanche, l'écart SVM vs Naive Bayes (2.8 pts)
dépasse l'écart-type, et là on peut raisonnablement conclure.
La validation croisée empêche de sur-interpréter de petits écarts qui
ne sont que du bruit d'échantillonnage.

--- PIÈGE CRITIQUE : LA FUITE DE DONNÉES ---

Il faut IMPÉRATIVEMENT utiliser un Pipeline (vectorizer + modèle) plutôt
que de vectoriser tout le dataset avant la CV. Sinon le vectorizer
calcule son vocabulaire ET ses poids IDF en "voyant" les données de test
de chaque fold -- une fuite (data leakage) qui gonfle artificiellement
les scores.

Vérifié empiriquement : sur un exemple jouet, l'IDF du mot "delivery"
vaut 1.8473 si le fit inclut le test, contre 1.5108 si le fit ne voit
que le train. Et le vocabulaire contient alors des mots exclusifs au
test. Le pipeline garantit que le vectorizer est re-entraîné uniquement
sur le train de chaque fold.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
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
