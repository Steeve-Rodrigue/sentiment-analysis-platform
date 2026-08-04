"""
src/classical_ml/lightgbm_model.py

Phase 4 — Machine Learning classique — bloc "LightGBM".

Théorie résumée :
LightGBM (Microsoft) est une autre implémentation optimisée du Gradient
Boosting, avec une différence algorithmique clé par rapport à XGBoost
dans la FAÇON DE FAIRE CROÎTRE LES ARBRES :

- XGBoost fait croître les arbres **par niveau** (level-wise) : il
  complète entièrement un niveau de profondeur avant de passer au
  suivant -- arbres équilibrés, mais on dépense du calcul sur des
  branches peu utiles.
- LightGBM fait croître **par feuille** (leaf-wise) : à chaque étape,
  il choisit LA feuille qui réduira le plus la perte et l'étend elle
  seule -- arbres déséquilibrés, mais chaque découpe est celle qui
  apporte le plus de gain.

Conséquence pratique : LightGBM converge généralement plus vite (moins
de découpes pour un même gain), d'où son nom "Light". Contrepartie :
les arbres déséquilibrés peuvent surapprendre plus facilement sur de
petits datasets -- d'où l'importance de num_leaves et min_child_samples.

Vérifié empiriquement sur movie_reviews (TF-IDF 2000 features) :
accuracy = 0.800, temps = 3.3s -- ~3x PLUS RAPIDE que XGBoost (9.4s)
mais 2.5 points moins précis (0.825). Le compromis vitesse/précision
est réel et mesurable ici.

Note technique : comme XGBoost, LightGBM exige des labels NUMÉRIQUES --
le LabelEncoder est encapsulé pour garder l'interface homogène.
"""

from __future__ import annotations

import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder


def train_lightgbm(
    X_train: list[str],
    y_train: list[str],
    max_features: int = 2000,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    num_leaves: int = 31,
    random_state: int = 42,
):
    """Vectorise en TF-IDF puis entraîne un LightGBM.

    num_leaves est LE paramètre clé de LightGBM (remplace max_depth) :
    il limite le nombre de feuilles par arbre. Comme la croissance est
    leaf-wise (déséquilibrée), c'est ce plafond qui contrôle la
    complexité, pas la profondeur -- augmenter num_leaves augmente
    rapidement le risque de surapprentissage."""
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)

    label_encoder = LabelEncoder()
    y_train_num = label_encoder.fit_transform(y_train)

    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        random_state=random_state,
        verbose=-1,
    )
    model.fit(X_train_vec, y_train_num)
    return model, vectorizer, label_encoder


def evaluate_model(
    model, vectorizer, label_encoder, X_test: list[str], y_test: list[str]
) -> dict:
    """Évalue un LightGBM entraîné, en redécodant les prédictions
    numériques en labels texte."""
    X_test_vec = vectorizer.transform(X_test)
    predictions_num = model.predict(X_test_vec)
    predictions = label_encoder.inverse_transform(predictions_num)
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "report": classification_report(y_test, predictions, output_dict=True),
        "predictions": predictions,
    }
