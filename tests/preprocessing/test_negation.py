"""
tests/preprocessing/test_negation.py

Tests unitaires pour src/preprocessing/negation.py — chaque test
correspond à un exemple vérifié manuellement pendant l'apprentissage.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from preprocessing.negation import mark_negation_scope


def test_marks_word_after_negation():
    tokens = ["delivery", "was", "not", "good"]
    result = mark_negation_scope(tokens)
    assert result == ["delivery", "was", "not", "good_NEG"]


def test_negation_scope_stops_at_punctuation():
    tokens = ["I", "do", "n't", "like", "this", ",", "but", "great", "service"]
    result = mark_negation_scope(tokens)
    assert "like_NEG" in result
    assert "great" in result  # après la virgule : pas marqué
    assert "great_NEG" not in result


def test_no_negation_leaves_tokens_untouched():
    tokens = ["the", "delivery", "was", "good"]
    result = mark_negation_scope(tokens)
    assert result == tokens  # rien ne doit changer, aucune négation présente


def test_marked_token_survives_stopword_removal():
    # démontre la correction du bug vu précédemment : remove_stopwords()
    # seul retirait "not" et inversait le sens ; ici, "good_NEG" ne
    # correspond à aucun stop-word donc survit intact.
    from preprocessing.normalization import remove_stopwords

    tokens = ["delivery", "was", "not", "good"]
    marked = mark_negation_scope(tokens)
    after_stopwords = remove_stopwords(marked)
    assert "good_NEG" in after_stopwords
