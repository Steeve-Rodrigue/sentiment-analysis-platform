"""
src/classical_ml/xgboost_model.py

Phase 4 — Machine Learning classique — bloc "XGBoost".

Théorie résumée :
XGBoost (eXtreme Gradient Boosting) est une réimplémentation optimisée
du Gradient Boosting classique (gradient_boosting_model.py). L'algorithme
de fond est le même (arbres séquentiels corrigeant les erreurs des
précédents), mais avec plusieurs améliorations :

1. **Régularisation intégrée** (L1 et L2 sur les poids des feuilles) --
   le Gradient Boosting de scikit-learn n'en a pas nativement, ce qui
   rend XGBoost moins sujet au surapprentissage.
2. **Parallélisation intelligente** : les arbres restent séquentiels
   (impossible de changer ça), mais la CONSTRUCTION de chaque arbre
   (recherche du meilleur point de découpe parmi des milliers de
   features) est parallélisée sur tous les cœurs.
3. **Gestion native des données creuses** (sparse) -- exactement notre
   cas avec du TF-IDF, où la majorité des cases sont à zéro.

Vérifié empiriquement sur movie_reviews (TF-IDF 2000 features) :
accuracy = 0.825, temps = 9.4s -- le MEILLEUR de tous les modèles à
base d'arbres de la Phase 4 (+3.3 points sur GradientBoosting et
RandomForest, tous deux à 0.792). Reste néanmoins derrière le SVM
linéaire (0.835), plus rapide de surcroît.

Note technique : XGBoost exige des labels NUMÉRIQUES (0/1), pas des
chaînes ("pos"/"neg") -- le LabelEncoder est encapsulé ici pour que
l'interface reste identique aux autres modèles de la phase.
"""

from __future__ import annotations

import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder


def train_xgboost(
    X_train: list[str],
    y_train: list[str],
    max_features: int = 2000,
    n_estimators: int = 100,
    learning_rate: float = 0.3,
    max_depth: int = 6,
    random_state: int = 42,
):
    """Vectorise en TF-IDF puis entraîne un XGBoost. Retourne
    (model, vectorizer, label_encoder) -- le label_encoder est
    nécessaire pour redécoder les prédictions numériques en "pos"/"neg"."""
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)

    label_encoder = LabelEncoder()
    y_train_num = label_encoder.fit_transform(y_train)

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
        verbosity=0,
    )
    model.fit(X_train_vec, y_train_num)
    return model, vectorizer, label_encoder


def evaluate_model(
    model, vectorizer, label_encoder, X_test: list[str], y_test: list[str]
) -> dict:
    """Évalue un XGBoost entraîné. Redécode automatiquement les
    prédictions numériques en labels texte d'origine."""
    X_test_vec = vectorizer.transform(X_test)
    predictions_num = model.predict(X_test_vec)
    predictions = label_encoder.inverse_transform(predictions_num)
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "report": classification_report(y_test, predictions, output_dict=True),
        "predictions": predictions,
    }
