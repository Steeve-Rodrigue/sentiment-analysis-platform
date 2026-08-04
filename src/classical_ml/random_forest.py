"""
src/classical_ml/random_forest_model.py

Phase 4 — Machine Learning classique — bloc "Random Forest".

Random Forest corrige le défaut principal de l'arbre unique
(decision_tree_model.py) : son instabilité. Le principe est le
**bagging** (bootstrap aggregating) :
1. Entraîner N arbres (100 par défaut) indépendamment, chacun sur un
   sous-échantillon ALÉATOIRE des données (tirage avec remise).
2. À chaque nœud, ne considérer qu'un sous-ensemble aléatoire des
   features (pas tous les mots) -- ce qui décorrèle les arbres entre eux.
3. Prédire par vote majoritaire des N arbres.

L'intuition : un arbre seul peut se tromper sur un cas particulier
(instabilité), mais il est peu probable que 100 arbres entraînés sur des
données/features différentes se trompent tous de la même façon. La
moyenne des erreurs individuelles s'annule partiellement.

Vérifié empiriquement sur movie_reviews (TF-IDF 2000 features) :
accuracy = 0.792, contre 0.682 pour l'arbre unique -- soit +11 points,
la plus grosse amélioration relative de toute la Phase 4. Le bagging
fonctionne exactement comme la théorie le prédit.

Prix à payer : on perd l'interprétabilité totale de l'arbre unique
(impossible de lire "le" chemin de décision -- il y en a 100), mais
feature_importances_ reste disponible (moyennée sur tous les arbres).
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report


def train_random_forest(
    X_train: list[str],
    y_train: list[str],
    max_features: int = 2000,
    n_estimators: int = 100,
    max_depth: int | None = None,
    random_state: int = 42,
):
    """Vectorise en TF-IDF puis entraîne une forêt aléatoire.
    n_estimators = nombre d'arbres dans la forêt (plus il y en a, plus
    la prédiction est stable, mais plus l'entraînement est lent).
    n_jobs=-1 utilise tous les cœurs disponibles -- les arbres étant
    indépendants, l'entraînement se parallélise parfaitement."""
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train_vec, y_train)
    return model, vectorizer


def evaluate_model(model, vectorizer, X_test: list[str], y_test: list[str]) -> dict:
    """Évalue un modèle déjà entraîné sur un jeu de test."""
    X_test_vec = vectorizer.transform(X_test)
    predictions = model.predict(X_test_vec)
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "report": classification_report(y_test, predictions, output_dict=True),
        "predictions": predictions,
    }
