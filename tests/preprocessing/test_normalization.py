"""
tests/preprocessing/test_normalization.py

Tests unitaires pour src/preprocessing/normalization.py — chaque test
correspond à un exemple vérifié manuellement pendant l'apprentissage.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from preprocessing.normalization import (
    lemmatize_tokens,
    remove_stopwords,
    stem_tokens,
)


def test_stem_tokens_reduces_common_suffixes():
    result = stem_tokens(["running", "runs"])
    assert result == ["run", "run"]


def test_stem_tokens_can_over_stem():
    # démontre une limite connue du stemming, volontairement testée
    # pour ne pas la cacher : "university" et "universal" fusionnent à tort
    result = stem_tokens(["university", "universal"])
    assert result[0] == result[1]  # comportement documenté, pas un bug caché


def test_lemmatize_without_pos_assumes_noun():
    # sans POS, "running" reste "running" (WordNet le traite comme un nom)
    result = lemmatize_tokens(["running"], use_pos=False)
    assert result == ["running"]


def test_lemmatize_with_pos_correctly_reduces_verb():
    tokens = [
        "The",
        "delivery",
        "was",
        "better",
        "than",
        "expected",
        "and",
        "arrived",
        "running",
    ]
    result = lemmatize_tokens(tokens, use_pos=True)
    assert "run" in result  # "running" (verbe) -> "run"
    assert "arrive" in result  # "arrived" -> "arrive"
    assert "good" in result  # "better" -> "good"


def test_remove_stopwords_drops_low_information_words():
    tokens = ["the", "delivery", "was", "good"]
    result = remove_stopwords(tokens)
    assert "delivery" in result
    assert "good" in result
    assert "the" not in result
    assert "was" not in result


def test_remove_stopwords_danger_removes_negation():
    # test volontaire qui documente le piège : "not" est un stop-word NLTK,
    # donc remove_stopwords() seule peut inverser le sens d'une phrase.
    tokens = ["delivery", "was", "not", "good"]
    result = remove_stopwords(tokens)
    assert "not" not in result  # comportement réel, dangereux pour le sentiment
