"""
tests/deep_learning/test_text_embedding.py

Tests unitaires pour src/deep_learning/text_embedding.py.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch

from deep_learning.text_embedding import (
    AverageEmbeddingClassifier,
    build_vocabulary,
    encode_texts,
    evaluate_classifier,
    train_classifier,
)


def test_vocabulary_includes_special_tokens():
    vocab = build_vocabulary(["the delivery was fast", "the product was great"])
    assert vocab["<pad>"] == 0
    assert vocab["<unk>"] == 1


def test_vocabulary_respects_max_size():
    textes = ["a b c d e f g h i j k l m n o p"] * 10
    vocab = build_vocabulary(textes, max_vocab_size=5)
    assert len(vocab) == 7  # 5 mots + <pad> + <unk>


def test_encode_texts_produces_fixed_length_tensor():
    vocab = build_vocabulary(["the delivery was fast"])
    encoded = encode_texts(
        ["the delivery", "the delivery was very fast indeed"], vocab, max_len=10
    )
    assert encoded.shape == (2, 10)


def test_encode_texts_pads_short_and_truncates_long():
    vocab = build_vocabulary(["a b c d e f g h"])
    court = encode_texts(["a b"], vocab, max_len=5)
    long = encode_texts(["a b c d e f g h"], vocab, max_len=5)
    assert (court[0, 2:] == 0).all()  # complete avec des zeros
    assert long.shape == (1, 5)  # tronque a 5, pas 8


def test_unknown_word_maps_to_unk_token():
    vocab = build_vocabulary(["the delivery was fast"])
    encoded = encode_texts(["completely_unknown_word_xyz"], vocab, max_len=3)
    assert encoded[0, 0].item() == 1  # <unk>


def test_padding_vector_stays_zero_after_training():
    vocab = {"<pad>": 0, "<unk>": 1, "good": 2, "bad": 3}
    model = AverageEmbeddingClassifier(vocab_size=len(vocab), embed_dim=4)

    X = torch.tensor([[2, 0, 0], [3, 0, 0]])
    y = torch.tensor([1.0, 0.0])
    train_classifier(model, X, y, epochs=5)

    assert torch.allclose(model.embedding.weight[0], torch.zeros(4))


def test_classifier_learns_something_on_toy_data():
    vocab = build_vocabulary(["good great excellent", "bad terrible awful"] * 20)
    X = encode_texts(
        ["good great excellent", "bad terrible awful"] * 20, vocab, max_len=5
    )
    y = torch.tensor(([1.0, 0.0] * 20))

    model = AverageEmbeddingClassifier(vocab_size=len(vocab), embed_dim=16)
    train_classifier(model, X, y, epochs=30, lr=0.01)
    results = evaluate_classifier(model, X, y)

    assert results["accuracy"] > 0.8  # doit apprendre ce pattern simple


def test_glove_initialized_embedding_differs_from_random():
    vocab = {"<pad>": 0, "<unk>": 1, "good": 2}
    matrice_glove = torch.zeros(len(vocab), 4)
    matrice_glove[2] = torch.tensor([1.0, 2.0, 3.0, 4.0])  # simule un vecteur GloVe

    model = AverageEmbeddingClassifier(
        vocab_size=len(vocab), embed_dim=4, pretrained_weights=matrice_glove
    )
    assert torch.allclose(model.embedding.weight[2], torch.tensor([1.0, 2.0, 3.0, 4.0]))
