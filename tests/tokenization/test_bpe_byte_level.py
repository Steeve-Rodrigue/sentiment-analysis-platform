"""
tests/tokenization/test_byte_level_bpe.py

Tests unitaires pour src/tokenization/byte_level_bpe.py — le test le
plus important vérifie la garantie centrale : aucun texte, dans aucune
langue/script, ne peut jamais produire un échec ou une perte de données.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tokenization.bpe_byte_level import (
    detokenize_with_byte_level_bpe,
    tokenize_with_byte_level_bpe,
    train_byte_level_bpe,
)

_CORPUS = [
    "the delivery was fast",
    "the delivery was slow",
    "the shipping was fast",
    "customer service was helpful",
] * 20


def _train_test_tokenizer():
    return train_byte_level_bpe(_CORPUS, vocab_size=300, min_frequency=2)


def test_known_word_reconstructs_exactly():
    tokenizer = _train_test_tokenizer()
    tokens = tokenize_with_byte_level_bpe(tokenizer, "delivery")
    reconstruit = detokenize_with_byte_level_bpe(tokenizer, tokens)
    assert reconstruit == "delivery"


def test_completely_unseen_ascii_reconstructs_exactly():
    # regression du piege initial_alphabet : sans lui, ce test echouerait
    tokenizer = _train_test_tokenizer()
    tokens = tokenize_with_byte_level_bpe(tokenizer, "xyzabc123")
    reconstruit = detokenize_with_byte_level_bpe(tokenizer, tokens)
    assert reconstruit == "xyzabc123"


def test_emoji_never_produces_unk_and_reconstructs_exactly():
    tokenizer = _train_test_tokenizer()
    tokens = tokenize_with_byte_level_bpe(tokenizer, "😡")
    assert "[UNK]" not in tokens
    reconstruit = detokenize_with_byte_level_bpe(tokenizer, tokens)
    assert reconstruit == "😡"


def test_japanese_script_never_seen_reconstructs_exactly():
    tokenizer = _train_test_tokenizer()
    tokens = tokenize_with_byte_level_bpe(tokenizer, "日本語")
    reconstruit = detokenize_with_byte_level_bpe(tokenizer, tokens)
    assert reconstruit == "日本語"
