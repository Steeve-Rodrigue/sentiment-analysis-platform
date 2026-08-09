"""
tests/deep_learning/test_text_attention.py

Tests unitaires pour src/deep_learning/text_attention.py.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch

from deep_learning.attention import LSTMWithAttention, get_attention_weights
from deep_learning.text_embedding import (
    build_vocabulary,
    encode_texts,
    evaluate_classifier,
    train_classifier,
)


def test_forward_pass_shape():
    model = LSTMWithAttention(vocab_size=50, embed_dim=8, hidden_dim=4)
    x = torch.randint(1, 50, (3, 10))
    assert model(x).shape == (3,)


def test_attention_weights_sum_to_one():
    model = LSTMWithAttention(vocab_size=50, embed_dim=8, hidden_dim=4)
    x = torch.randint(1, 50, (2, 6))
    _, weights = model(x, return_weights=True)
    sums = weights.sum(dim=1)
    assert torch.allclose(sums, torch.ones(2), atol=1e-5)


def test_padding_gets_exactly_zero_attention():
    model = LSTMWithAttention(vocab_size=50, embed_dim=8, hidden_dim=4)
    # 3 vrais tokens puis 2 positions de padding (id 0)
    x = torch.tensor([[5, 10, 15, 0, 0]])
    _, weights = model(x, return_weights=True)
    assert torch.allclose(weights[0, 3:], torch.zeros(2), atol=1e-6)


def test_get_attention_weights_helper_matches_forward():
    model = LSTMWithAttention(vocab_size=50, embed_dim=8, hidden_dim=4)
    x = torch.randint(1, 50, (2, 6))
    weights_helper = get_attention_weights(model, x)
    with torch.no_grad():
        _, weights_direct = model(x, return_weights=True)
    assert torch.allclose(weights_helper, weights_direct)


def test_padding_embedding_stays_zero_after_training():
    vocab = build_vocabulary(["good great excellent", "bad terrible awful"] * 20)
    texts = ["good great excellent", "bad terrible awful"] * 20
    X = encode_texts(texts, vocab, max_len=5)
    y = torch.tensor(([1.0, 0.0] * 20))

    model = LSTMWithAttention(vocab_size=len(vocab), embed_dim=8, hidden_dim=4)
    train_classifier(model, X, y, epochs=5)

    assert torch.allclose(model.embedding.weight[0], torch.zeros(8))


def test_model_learns_and_focuses_on_sentiment_word():
    # verifie empiriquement le resultat cle observe dans la conversation :
    # le mot porteur de sentiment doit recevoir le plus de poids
    torch.manual_seed(42)
    phrases = ["the delivery was terrible", "the delivery was excellent"] * 30
    vocab = build_vocabulary(phrases, max_vocab_size=20)
    X = encode_texts(phrases, vocab, max_len=5)
    y = torch.tensor(([0.0, 1.0] * 30))

    model = LSTMWithAttention(vocab_size=len(vocab), embed_dim=16, hidden_dim=16)
    train_classifier(model, X, y, epochs=100, lr=0.01)

    results = evaluate_classifier(model, X, y)
    assert results["accuracy"] > 0.85

    test_phrase = encode_texts(["the delivery was terrible"], vocab, max_len=5)
    weights = get_attention_weights(model, test_phrase)

    # "terrible" est le 4eme mot (index 3) -- doit dominer l'attention
    poids_terrible = weights[0, 3].item()
    assert poids_terrible > 0.4
