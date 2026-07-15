"""
src/preprocessing/negation.py

Phase 1 — NLP Fundamentals — bloc "négation".

Théorie résumée (voir la conversation associée pour le détail) :
Un mot de négation ("not", "never", "n't"...) inverse la polarité des
mots qui le suivent, jusqu'à la prochaine ponctuation. Sans marquage
explicite, un modèle Bag-of-Words voit "good" dans "not good" comme un
signal positif, alors que le sens réel est négatif.

La technique retenue ici (suffixe "_NEG") transforme "good" en "good_NEG"
-- un token distinct, que des modèles classiques (Phase 4) peuvent
apprendre à associer au sentiment négatif sans comprendre la grammaire.

IMPORTANT (ordre du pipeline) : cette fonction doit s'exécuter AVANT
remove_stopwords() (normalization.py). Une fois qu'un mot est marqué
"good_NEG", il ne correspond plus à aucune entrée de la liste de
stop-words, donc il survit à la suppression -- ce qui corrige
précisément le bug observé plus tôt (remove_stopwords() seul retire
"not" et inverse silencieusement le sens de la phrase).
"""

from __future__ import annotations

NEGATION_WORDS = {"not", "no", "never", "n't", "cannot", "nothing", "neither", "nor"}
PUNCTUATION = {".", "!", "?", ",", ";", ":"}


def mark_negation_scope(tokens: list[str]) -> list[str]:
    """Ajoute le suffixe "_NEG" à chaque token situé entre un mot de
    négation et la ponctuation suivante (la portée de la négation
    s'arrête à la ponctuation, pas à la fin de la phrase entière)."""
    result = []
    negating = False
    for token in tokens:
        if token.lower() in NEGATION_WORDS:
            result.append(token)
            negating = True
        elif token in PUNCTUATION:
            result.append(token)
            negating = False
        elif negating:
            result.append(token + "_NEG")
        else:
            result.append(token)
    return result
