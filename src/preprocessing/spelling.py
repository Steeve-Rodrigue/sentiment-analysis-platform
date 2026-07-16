"""
src/preprocessing/spelling.py

Phase 1 — NLP Fundamentals — bloc "correction orthographique".

Théorie résumée (voir la conversation associée pour le détail) :
Un correcteur orthographique compare un mot inconnu à un dictionnaire et
choisit le mot le plus proche selon la distance d'édition (Levenshtein --
nombre minimal d'insertions/suppressions/substitutions pour transformer
un mot en un autre), en départageant les égalités par fréquence d'usage
("grat" est aussi proche de "great" que de "grant" en distance, mais
"great" est bien plus fréquent).

Même objectif que la lemmatisation (normalization.py) : regrouper des
variantes de surface d'un même mot réel, ici causées par des fautes de
frappe plutôt que par la grammaire -- sans ça, "delivery" et "deliery"
sont deux tokens sans aucun lien pour un modèle.

Garde-fou : ne corrige QUE les mots absents du dictionnaire, et ne
remplace jamais si aucune correction fiable n'est trouvée (spell.correction
retourne None) -- ça protège les noms de marque/produit (ex. "Samsung")
d'être "corrigés" à tort en un mot proche mais non pertinent.

Ordre dans le pipeline : à exécuter tôt, juste après la tokenisation,
avant la négation/stop-words/lemmatisation -- un mot mal orthographié ne
correspondrait correctement à aucune de ces étapes suivantes non plus.
"""

from __future__ import annotations

from spellchecker import SpellChecker

_SPELL = SpellChecker()


def correct_spelling(tokens: list[str]) -> list[str]:
    """Corrige les tokens absents du dictionnaire. Garde le token original
    si aucune correction fiable n'est trouvée (protège les mots inconnus
    mais légitimes comme les noms de marque)."""
    corrected = []
    for token in tokens:
        if token.lower() in _SPELL.unknown([token.lower()]):
            correction = _SPELL.correction(token.lower())
            corrected.append(correction if correction else token)
        else:
            corrected.append(token)
    return corrected
