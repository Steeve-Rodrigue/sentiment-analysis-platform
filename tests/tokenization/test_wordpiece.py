"""
tests/tokenization/test_wordpiece_tokenizer.py

Tests unitaires pour src/tokenization/wordpiece_tokenizer.py — chaque
test correspond à un exemple vérifié manuellement pendant l'apprentissage.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tokenization.wordpiece import tokenize_with_wordpiece, train_wordpiece_tokenizer

_CORPUS = [
    "the delivery was fast",
    "the delivery was slow",
    "the shipping was fast",
    "the shipping was slow",
    "the product quality was great",
    "the product quality was poor",
    "customer service was helpful",
    "customer service was rude",
    "the packaging was damaged",
    "the packaging was excellent",
] * 20


def _train_test_tokenizer():
    return train_wordpiece_tokenizer(_CORPUS, vocab_size=100, min_frequency=2)


def test_continuation_fragments_are_prefixed_with_double_hash():
    tokenizer = _train_test_tokenizer()
    tokens = tokenize_with_wordpiece(tokenizer, "deliveryman")
    # tous les fragments SAUF le tout premier doivent porter le prefixe "##"
    assert not tokens[0].startswith("##")
    assert all(tok.startswith("##") for tok in tokens[1:])


def test_frequent_word_stays_compact():
    tokenizer = _train_test_tokenizer()
    tokens = tokenize_with_wordpiece(tokenizer, "delivery")
    # doit rester peu fragmente (mot frequent dans le corpus d'entrainement)
    assert len(tokens) <= 5


def test_completely_foreign_word_never_fully_fails():
    tokenizer = _train_test_tokenizer()
    tokens = tokenize_with_wordpiece(tokenizer, "xyzabc123")
    assert len(tokens) > 0


def test_vocab_size_is_respected():
    tokenizer = train_wordpiece_tokenizer(_CORPUS, vocab_size=50, min_frequency=2)
    assert tokenizer.get_vocab_size() <= 50
