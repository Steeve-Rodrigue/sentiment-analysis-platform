"""
src/classical_ml/naive_bayes_model.py

Phase 4 — Machine Learning classique — bloc "Naive Bayes".

Naive Bayes applique le théorème de Bayes en supposant (à tort, mais de
façon utile) que chaque mot est indépendant des autres sachant la
classe :
    P(classe | texte) ∝ P(classe) * produit(P(mot_i | classe))
Cette hypothèse "naïve" rend le calcul très simple et rapide, malgré
son inexactitude linguistique évidente (les mots d'une phrase ne sont
clairement pas indépendants).

Vérifié empiriquement sur le vrai corpus movie_reviews (2000 avis,
Pang & Lee) : ~80% d'accuracy, précision/rappel équilibrés entre les
deux classes -- un score honnête pour un premier modèle simple, qui
sert de ligne de base pour comparer les modèles suivants de cette phase.

Note technique : MultinomialNB attend des comptages/fréquences non
négatifs (ce que TF-IDF produit) -- ne fonctionnerait pas directement
sur des embeddings denses type Word2Vec, qui peuvent être négatifs.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB


def train_naive_bayes(X_train: list[str], y_train: list[str], max_features: int = 5000):
    """Vectorise en TF-IDF puis entraîne un classifieur Naive Bayes
    multinomial. Retourne le modèle ET le vectorizer (nécessaire pour
    transformer les données de test de façon cohérente)."""
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    return model, vectorizer


def evaluate_model(model, vectorizer, X_test: list[str], y_test: list[str]) -> dict:
    """Évalue un modèle déjà entraîné sur un jeu de test. Retourne
    l'accuracy et un rapport détaillé (précision/rappel/F1 par classe)."""
    X_test_vec = vectorizer.transform(X_test)
    predictions = model.predict(X_test_vec)
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "report": classification_report(y_test, predictions, output_dict=True),
        "predictions": predictions,
    }
