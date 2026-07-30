"""
tests/tokenization/test_sentencepiece_tokenizer.py

Tests unitaires pour src/tokenization/sentencepiece_tokenizer.py —
chaque test correspond à un exemple vérifié manuellement pendant
l'apprentissage.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tokenization.sentencepiece import (
    detokenize_with_sentencepiece,
    tokenize_with_sentencepiece,
    train_sentencepiece_tokenizer,
)

_CORPUS_LINES = [
    "the delivery was fast",
    "the delivery was slow",
    "la livraison etait rapide",
    "la livraison etait lente",
    "the product quality was great",
    "la qualite du produit est excellente",
] * 30


def _train_test_processor(tmp_path_prefix):
    corpus_file = tmp_path_prefix + "_corpus.txt"
    with open(corpus_file, "w", encoding="utf-8") as f:
        f.write("\n".join(_CORPUS_LINES))
    return train_sentencepiece_tokenizer(
        corpus_file_path=corpus_file,
        model_prefix=tmp_path_prefix + "_model",
        vocab_size=100,
        model_type="bpe",
    )


def test_encode_decode_is_reversible_on_clean_text():
    with tempfile.TemporaryDirectory() as tmp_dir:
        processor = _train_test_processor(os.path.join(tmp_dir, "sp"))
        texte = "the delivery was fast"
        tokens = tokenize_with_sentencepiece(processor, texte)
        reconstruit = detokenize_with_sentencepiece(processor, tokens)
        assert reconstruit == texte


def test_space_is_encoded_as_meta_symbol():
    with tempfile.TemporaryDirectory() as tmp_dir:
        processor = _train_test_processor(os.path.join(tmp_dir, "sp"))
        tokens = tokenize_with_sentencepiece(processor, "the delivery")
        # le symbole meta "▁" doit apparaitre, marquant les espaces
        assert any("▁" in tok for tok in tokens)


def test_works_identically_on_different_language_without_code_change():
    with tempfile.TemporaryDirectory() as tmp_dir:
        processor = _train_test_processor(os.path.join(tmp_dir, "sp"))
        texte_fr = "la livraison etait rapide"
        tokens = tokenize_with_sentencepiece(processor, texte_fr)
        reconstruit = detokenize_with_sentencepiece(processor, tokens)
        assert reconstruit == texte_fr
        assert len(tokens) > 0
