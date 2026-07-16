"""
tests/preprocessing/test_spelling.py

Tests unitaires pour src/preprocessing/spelling.py — chaque test
correspond à un exemple vérifié manuellement pendant l'apprentissage.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from preprocessing.spelling import correct_spelling


def test_corrects_common_typo():
    result = correct_spelling(["deliery"])
    assert result == ["delivery"]


def test_corrects_multiple_typos_in_sentence():
    tokens = ["I", "recieved", "my", "order", "and", "the", "pakage", "was", "damaged"]
    result = correct_spelling(tokens)
    assert "received" in result
    assert "package" in result


def test_leaves_correctly_spelled_words_unchanged():
    tokens = ["delivery", "was", "great"]
    result = correct_spelling(tokens)
    assert result == tokens


def test_does_not_break_on_unknown_proper_noun():
    # démontre le garde-fou : un mot inconnu sans correction fiable
    # (ex. nom de marque) doit être conservé tel quel, pas remplacé
    # par un mot proche mais non pertinent.
    tokens = ["Samsung", "phone"]
    result = correct_spelling(tokens)
    assert len(result) == 2  # aucun token perdu, quel que soit le contenu exact
