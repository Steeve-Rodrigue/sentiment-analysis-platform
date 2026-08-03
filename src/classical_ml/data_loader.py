"""
src/classical_ml/data_loader.py

Phase 4 — Machine Learning classique — chargement du dataset partagé.

Tous les modèles de cette phase (Naive Bayes, Logistic Regression, SVM,
arbres, Random Forest, XGBoost, LightGBM, CatBoost) s'entraînent et se
comparent sur le MÊME dataset réel -- le corpus movie_reviews de NLTK
(Pang & Lee, 2000 avis de films, 1000 positifs / 1000 négatifs) --
plutôt que chacun sur des données différentes, pour que les
comparaisons de métriques entre modèles aient un sens.
"""

from __future__ import annotations

from nltk.corpus import movie_reviews
from sklearn.model_selection import train_test_split


def load_movie_reviews(test_size: float = 0.2, random_state: int = 42):
    """Charge le corpus movie_reviews et retourne un split
    train/test déjà stratifié (même proportion pos/neg dans les deux
    ensembles)."""
    documents = [
        (movie_reviews.raw(fileid), category)
        for category in movie_reviews.categories()
        for fileid in movie_reviews.fileids(category)
    ]
    texts = [texte for texte, label in documents]
    labels = [label for texte, label in documents]

    return train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
