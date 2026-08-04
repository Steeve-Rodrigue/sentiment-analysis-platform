"""
src/classical_ml/gradient_boosting_model.py

Phase 4 — Machine Learning classique — bloc "Gradient Boosting".

Théorie résumée :
Gradient Boosting change radicalement de stratégie par rapport au
Random Forest. Là où le bagging entraîne N arbres INDÉPENDAMMENT en
parallèle puis vote, le boosting entraîne les arbres SÉQUENTIELLEMENT :

1. Entraîner un premier arbre (volontairement faible, peu profond).
2. Mesurer ses erreurs sur les données d'entraînement.
3. Entraîner un DEUXIÈME arbre spécialisé pour corriger précisément
   ces erreurs.
4. Répéter -- chaque nouvel arbre s'attaque aux résidus laissés par
   l'ensemble des précédents.

Le nom "gradient" vient de la formalisation mathématique : chaque
nouvel arbre approxime le gradient (la direction de descente) de la
fonction de perte, comme une descente de gradient classique mais où
chaque "pas" est un arbre entier.

CONSÉQUENCE IMPORTANTE : contrairement au bagging, ajouter des arbres
au boosting PEUT provoquer du surapprentissage -- chaque arbre
supplémentaire colle davantage aux données d'entraînement. Le nombre
d'arbres (n_estimators) et le learning_rate sont donc des
hyperparamètres critiques, pas juste "plus = mieux".

Vérifié empiriquement sur movie_reviews (TF-IDF 2000 features) :
accuracy = 0.792, temps = 5.9s -- même score que Random Forest (0.792)
mais 4x plus lent, car les arbres ne peuvent PAS être parallélisés
(chacun dépend du précédent). Les implémentations optimisées XGBoost /
LightGBM / CatBoost existent précisément pour corriger cette lenteur.
"""

from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report


def train_gradient_boosting(
    X_train: list[str],
    y_train: list[str],
    max_features: int = 2000,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    random_state: int = 42,
):
    """Vectorise en TF-IDF puis entraîne un Gradient Boosting.

    learning_rate contrôle l'ampleur de la correction apportée par
    chaque nouvel arbre : petit = corrections prudentes (besoin de plus
    d'arbres, mais moins de surapprentissage), grand = corrections
    agressives (converge vite, risque de surapprendre).

    max_depth=3 par défaut : contrairement au Random Forest (arbres
    profonds), le boosting utilise volontairement des arbres FAIBLES
    ("weak learners") -- c'est leur accumulation séquentielle qui fait
    la force, pas leur puissance individuelle."""
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
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
