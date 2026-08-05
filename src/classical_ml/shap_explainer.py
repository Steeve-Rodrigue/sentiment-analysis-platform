"""
src/classical_ml/shap_explainer.py

Phase 4 — Machine Learning classique — bloc "SHAP (explicabilité)".

Jusqu'ici on savait quels mots comptent EN GÉNÉRAL pour un modèle
(via coef_ ou feature_importances_). SHAP répond à une question plus
fine : "pour CET avis précis, quels mots ont poussé la prédiction, et
de combien ?"

Le concept vient de la théorie des jeux (valeurs de Shapley, Nobel
d'économie 1972) : dans une équipe qui gagne, comment répartir le gain
selon la contribution réelle de chaque joueur ? Ici chaque MOT est un
joueur, la PRÉDICTION est le gain à répartir. La valeur SHAP d'un mot
est sa contribution marginale moyenne sur toutes les combinaisons
possibles de mots présents/absents.

PROPRIÉTÉ CLÉ (additivité) : valeur_de_base + somme(valeurs SHAP) =
exactement le score du modèle. Vérifié empiriquement : -1.3589 des deux
côtés sur un avis de test. Rien ne se perd, la décomposition est
complète -- c'est ce qui distingue SHAP d'heuristiques approximatives.

PIÈGE D'INTERPRÉTATION IMPORTANT, observé empiriquement : le mot "bad"
peut avoir une contribution POSITIVE (+0.09) dans un avis négatif.
Ce n'est pas un bug : SHAP mesure la contribution RELATIVE À LA MOYENNE.
Si le modèle s'attend à voir "bad" très fréquemment dans les avis
négatifs et que son score TF-IDF est ici PLUS FAIBLE que cette moyenne,
sa présence relative pousse vers le positif.

USAGE POUR L'AUDIT DE BIAIS (Phase 13) : sur un avis test, SHAP a
révélé "eddie" et "murphy" (nom d'un acteur) parmi les contributeurs
négatifs -- le même biais de dataset qu'on avait repéré globalement
avec Naive Bayes ("seagal", "schumacher"), mais cette fois visible au
niveau d'une prédiction individuelle.

Choix d'explainer : LinearExplainer pour les modèles linéaires (forme
close, rapide), TreeExplainer pour les modèles à arbres (algorithme
exact et rapide adapté aux arbres). KernelExplainer existe pour les
modèles quelconques mais est BEAUCOUP plus lent (approximation par
échantillonnage) -- non utilisé ici.
"""

from __future__ import annotations

import numpy as np
import shap


def build_linear_explainer(model, X_train_vec, max_samples: int = 100):
    """Construit un explainer SHAP pour un modèle LINÉAIRE
    (LogisticRegression, LinearSVC). Utilise une forme close, donc
    rapide -- pas d'approximation par échantillonnage.

    max_samples limite la taille du jeu de référence ("background")
    utilisé pour estimer la distribution moyenne des features."""
    masker = shap.maskers.Independent(X_train_vec, max_samples=max_samples)
    return shap.LinearExplainer(model, masker)


def build_tree_explainer(model):
    """Construit un explainer SHAP pour un modèle À ARBRES
    (DecisionTree, RandomForest, XGBoost, LightGBM, CatBoost).
    TreeExplainer utilise un algorithme exact spécifique aux arbres,
    bien plus rapide que l'approche générique."""
    return shap.TreeExplainer(model)


def explain_prediction(
    explainer,
    X_vec,
    feature_names,
    index: int = 0,
    top_n: int = 10,
    class_index: int = 1,
) -> dict:
    """Explique UNE prédiction précise : retourne les top_n mots ayant
    le plus contribué (en valeur absolue), avec leur contribution signée.

    Une contribution positive pousse vers la classe positive, négative
    vers la classe négative -- mais attention au piège documenté en
    en-tête : c'est relatif à la MOYENNE, pas au sens intuitif du mot.

    PIÈGE TECHNIQUE géré ici : les explainers ne retournent PAS tous la
    même forme. LinearExplainer donne (nb_avis, nb_features) ;
    TreeExplainer donne (nb_avis, nb_features, nb_classes) -- trois
    dimensions. Certaines versions retournent aussi une liste, une
    entrée par classe. class_index sélectionne la classe à expliquer
    (1 = classe positive par défaut)."""
    shap_values = explainer.shap_values(X_vec)

    # certains explainers retournent une liste (une entree par classe)
    if isinstance(shap_values, list):
        shap_values = shap_values[class_index]

    shap_values = np.asarray(shap_values)

    # TreeExplainer : (nb_avis, nb_features, nb_classes) -> on selectionne la classe
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, class_index]

    valeurs = shap_values[index]

    # ne garder que les mots reellement presents (valeur non nulle)
    non_nuls = np.where(valeurs != 0)[0]
    base = explainer.expected_value
    base = float(
        base[class_index] if hasattr(base, "__len__") and len(base) > 1 else base
    )

    if len(non_nuls) == 0:
        return {
            "contributions": [],
            "base_value": base,
            "sum_of_all_contributions": 0.0,
            "reconstructed_score": base,
        }

    ordre = non_nuls[np.argsort(np.abs(valeurs[non_nuls]))[::-1]][:top_n]

    contributions = [
        {
            "word": str(feature_names[i]),
            "contribution": float(valeurs[i]),
            "pushes_toward": "positive" if valeurs[i] > 0 else "negative",
        }
        for i in ordre
    ]

    return {
        "contributions": contributions,
        "base_value": base,
        "sum_of_all_contributions": float(valeurs.sum()),
        "reconstructed_score": base + float(valeurs.sum()),
    }


def verify_additivity(
    explanation: dict, actual_model_score: float, tolerance: float = 1e-4
) -> bool:
    """Vérifie la propriété d'additivité de SHAP : base + somme des
    contributions doit égaler le score réel du modèle.

    Utile comme test de sanité -- si cette égalité échoue, c'est que
    l'explainer ou le modèle n'ont pas été correctement appariés."""
    return abs(explanation["reconstructed_score"] - actual_model_score) < tolerance


def format_explanation(explanation: dict) -> str:
    """Formate une explication en texte lisible."""
    lignes = [f"Valeur de base : {explanation['base_value']:+.4f}", ""]
    for c in explanation["contributions"]:
        fleche = "-> pos" if c["contribution"] > 0 else "-> neg"
        lignes.append(f"  {c['word']:15} {c['contribution']:+.4f}  {fleche}")
    lignes.append("")
    lignes.append(f"Score reconstruit : {explanation['reconstructed_score']:+.4f}")
    return "\n".join(lignes)
