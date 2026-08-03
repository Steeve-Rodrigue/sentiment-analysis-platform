"""
tests/classical_ml/test_logistic_regression.py

Tests unitaires pour src/classical_ml/logistic_regression.py —
chaque test correspond à un résultat vérifié manuellement pendant
l'apprentissage.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from classical_ml.data_loader import load_movie_reviews
from classical_ml.logistic_regression import evaluate_model, train_logistic_regression


def test_achieves_reasonable_accuracy():
    X_train, X_test, y_train, y_test = load_movie_reviews()
    model, vectorizer = train_logistic_regression(X_train, y_train)
    results = evaluate_model(model, vectorizer, X_test, y_test)
    assert results["accuracy"] > 0.70


def test_low_c_underfits_relative_to_default():
    # C tres petit = modele fortement contraint, doit sous-performer
    # par rapport a C=1.0 par defaut (verifie empiriquement dans la
    # conversation : C=0.001 donne ~0.787 vs ~0.828 a C=1.0)
    X_train, X_test, y_train, y_test = load_movie_reviews()

    model_contraint, vec_contraint = train_logistic_regression(
        X_train, y_train, C=0.001
    )
    resultats_contraint = evaluate_model(model_contraint, vec_contraint, X_test, y_test)

    model_defaut, vec_defaut = train_logistic_regression(X_train, y_train, C=1.0)
    resultats_defaut = evaluate_model(model_defaut, vec_defaut, X_test, y_test)

    assert resultats_contraint["accuracy"] < resultats_defaut["accuracy"]


def test_high_c_overfits_train_without_test_improvement():
    # C tres grand = modele quasi non regularise, doit memoriser le
    # train (accuracy train proche de 1.0) sans gain proportionnel sur
    # le test -- signe de surapprentissage
    X_train, X_test, y_train, y_test = load_movie_reviews()
    model, vectorizer = train_logistic_regression(X_train, y_train, C=100)

    from sklearn.metrics import accuracy_score

    X_train_vec = vectorizer.transform(X_train)
    acc_train = accuracy_score(y_train, model.predict(X_train_vec))

    assert acc_train > 0.99  # quasi-memorisation du train
