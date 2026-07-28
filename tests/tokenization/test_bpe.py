"""
tests/tokenization/test_bpe_tokenizer.py

Tests unitaires pour src/tokenization/bpe_tokenizer.py — chaque test
correspond à un exemple vérifié manuellement pendant l'apprentissage.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tokenization.bpe import tokenize_with_bpe, train_bpe_tokenizer

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
    return train_bpe_tokenizer(_CORPUS, vocab_size=100, min_frequency=2)


def test_frequent_word_becomes_single_token():
    tokenizer = _train_test_tokenizer()
    tokens = tokenize_with_bpe(tokenizer, "delivery")
    assert tokens == ["delivery"]


def test_unseen_word_reuses_known_fragment():
    # "deliveryman" n'existe dans aucune phrase du corpus, mais partage
    # le fragment "delivery" avec des mots connus
    tokenizer = _train_test_tokenizer()
    tokens = tokenize_with_bpe(tokenizer, "deliveryman")
    assert "delivery" in tokens
    assert len(tokens) > 1  # decoupe en plusieurs fragments, pas un echec


def test_completely_foreign_word_never_fully_fails():
    # meme un mot totalement etranger au corpus produit une sortie
    # (fragments/caracteres/[UNK]), jamais une exception
    tokenizer = _train_test_tokenizer()
    tokens = tokenize_with_bpe(tokenizer, "xyzabc123")
    assert len(tokens) > 0


def test_vocab_size_is_respected():
    tokenizer = train_bpe_tokenizer(_CORPUS, vocab_size=50, min_frequency=2)
    assert tokenizer.get_vocab_size() <= 50
