"""
tests/preprocessing/test_tokenization.py

Tests unitaires pour src/preprocessing/tokenization.py — chaque test
correspond à un exemple vérifié manuellement pendant l'apprentissage.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from preprocessing.tokenization import (
    rule_based_tokenize,
    sentence_tokenize,
    tokenize_document,
    whitespace_tokenize,
)


def test_whitespace_tokenize_keeps_punctuation_attached():
    tokens = whitespace_tokenize("The delivery was great!")
    assert "great!" in tokens  # ponctuation collée, comportement attendu
    assert "great" not in tokens


def test_rule_based_tokenize_splits_punctuation():
    tokens = rule_based_tokenize("The delivery was great!")
    assert "great" in tokens
    assert "!" in tokens
    assert "great!" not in tokens


def test_rule_based_tokenize_splits_contractions():
    tokens = rule_based_tokenize("I don't regret it.")
    assert "do" in tokens
    assert "n't" in tokens
    assert "don't" not in tokens


def test_sentence_tokenize_handles_abbreviations():
    text = "Dr. Smith recommended this product. It was excellent."
    sentences = sentence_tokenize(text)
    assert len(sentences) == 2
    assert sentences[0] == "Dr. Smith recommended this product."
    assert sentences[1] == "It was excellent."


def test_sentence_tokenize_naive_split_would_have_failed():
    # démontre pourquoi sentence_tokenize existe : un split naïf sur '.'
    # casserait "Dr." en un fragment séparé, ce que sentence_tokenize évite
    text = "Dr. Smith recommended it."
    naive_result = text.split(".")
    correct_result = sentence_tokenize(text)
    assert len(naive_result) > len(correct_result)  # le naïf sur-découpe
    assert len(correct_result) == 1


def test_tokenize_document_returns_list_of_lists():
    text = "The delivery was great! The packaging was poor."
    result = tokenize_document(text)
    assert len(result) == 2  # 2 phrases
    assert isinstance(result[0], list)  # chaque phrase = liste de tokens
    assert "packaging" in result[1]
    assert "poor" in result[1]
