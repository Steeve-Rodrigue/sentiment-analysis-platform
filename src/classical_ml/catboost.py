"""
src/classical_ml/catboost_model.py

Phase 4 — Machine Learning classique — bloc "CatBoost".

Théorie résumée :
CatBoost (Yandex) est la troisième grande implémentation optimisée du
Gradient Boosting, avec deux particularités algorithmiques :

1. **Arbres symétriques (oblivious trees)** : à chaque niveau de
   profondeur, TOUS les nœuds utilisent la MÊME condition de découpe.
   Cela produit des arbres parfaitement équilibrés -- plus rapides à
   évaluer en inférence et moins sujets au surapprentissage, au prix
   d'une expressivité réduite par arbre.
2. **Ordered boosting** : une technique pour éviter le "target leakage"
   (fuite d'information de la cible dans les statistiques calculées sur
   les données d'entraînement) -- le nom "Cat" vient de "Categorical",
   car CatBoost excelle surtout sur les variables catégorielles brutes,
   qu'il encode intelligemment sans qu'on ait à le faire soi-même.

LIMITE MAJEURE VÉRIFIÉE EMPIRIQUEMENT sur movie_reviews (TF-IDF 2000
features) : accuracy = 0.823, temps = 220 SECONDES -- soit ~150x plus
lent que Random Forest (1.5s) et ~23x plus lent que XGBoost (9.4s),
pour un score légèrement INFÉRIEUR à XGBoost (0.825).

Sur ce projet, CatBoost n'est donc PAS le bon choix : son avantage
principal (gestion native des variables catégorielles) ne sert à rien
ici, puisque nos features TF-IDF sont déjà numériques et creuses. Il est
inclus par exhaustivité pédagogique et pour documenter ce constat --
pas parce qu'il est recommandé pour ce cas d'usage.
"""

from __future__ import annotations

from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder


def train_catboost(
    X_train: list[str],
    y_train: list[str],
    max_features: int = 2000,
    iterations: int = 100,
    learning_rate: float = 0.1,
    depth: int = 6,
    random_state: int = 42,
):
    """Vectorise en TF-IDF puis entraîne un CatBoost.

    ATTENTION : très lent sur ce type de données (~220s avec les
    paramètres par défaut sur movie_reviews). Réduire `iterations` ou
    `max_features` pour des essais rapides.

    `iterations` est le nom CatBoost de `n_estimators` (nombre d'arbres),
    et `depth` celui de `max_depth` -- vocabulaire différent des autres
    bibliothèques, même concept."""
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)

    label_encoder = LabelEncoder()
    y_train_num = label_encoder.fit_transform(y_train)

    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        random_state=random_state,
        verbose=0,
    )
    model.fit(X_train_vec, y_train_num)
    return model, vectorizer, label_encoder


def evaluate_model(
    model, vectorizer, label_encoder, X_test: list[str], y_test: list[str]
) -> dict:
    """Évalue un CatBoost entraîné, en redécodant les prédictions
    numériques en labels texte."""
    X_test_vec = vectorizer.transform(X_test)
    predictions_num = model.predict(X_test_vec)
    predictions = label_encoder.inverse_transform(predictions_num.ravel().astype(int))
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "report": classification_report(y_test, predictions, output_dict=True),
        "predictions": predictions,
    }
