"""
tests/deep_learning/test_rnn_models.py

Tests unitaires pour src/deep_learning/rnn_models.py.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch

from deep_learning.RNN import TextBiLSTM, TextGRU, TextLSTM
from deep_learning.text_embedding import (
    build_vocabulary,
    encode_texts,
    evaluate_classifier,
    train_classifier,
)


def test_lstm_forward_pass_shape():
    model = TextLSTM(vocab_size=50, embed_dim=8, hidden_dim=4)
    x = torch.randint(1, 50, (3, 10))
    assert model(x).shape == (3,)


def test_bilstm_uses_double_hidden_dim_in_final_layer():
    # bidirectional=True concatene 2 directions -> la couche finale doit
    # attendre hidden_dim * 2 en entree
    model = TextBiLSTM(vocab_size=50, embed_dim=8, hidden_dim=4)
    assert model.fc.in_features == 8  # 4 * 2


def test_bilstm_forward_pass_shape():
    model = TextBiLSTM(vocab_size=50, embed_dim=8, hidden_dim=4)
    x = torch.randint(1, 50, (3, 10))
    assert model(x).shape == (3,)


def test_gru_forward_pass_shape():
    model = TextGRU(vocab_size=50, embed_dim=8, hidden_dim=4)
    x = torch.randint(1, 50, (3, 10))
    assert model(x).shape == (3,)


def test_gru_has_fewer_parameters_than_lstm_same_size():
    # verifie la theorie : GRU (2 portes) a moins de parametres que
    # LSTM (3 portes) a hidden_dim egal
    lstm = TextLSTM(vocab_size=50, embed_dim=8, hidden_dim=16)
    gru = TextGRU(vocab_size=50, embed_dim=8, hidden_dim=16)

    params_lstm = sum(p.numel() for p in lstm.lstm.parameters())
    params_gru = sum(p.numel() for p in gru.gru.parameters())

    assert params_gru < params_lstm


def test_padding_vector_stays_zero_after_training_lstm():
    vocab = build_vocabulary(["good great excellent", "bad terrible awful"] * 20)
    X = encode_texts(
        ["good great excellent", "bad terrible awful"] * 20, vocab, max_len=5
    )
    y = torch.tensor(([1.0, 0.0] * 20))

    model = TextLSTM(vocab_size=len(vocab), embed_dim=8, hidden_dim=4)
    train_classifier(model, X, y, epochs=5)

    assert torch.allclose(model.embedding.weight[0], torch.zeros(8))


def test_lstm_learns_something_on_toy_data():
    vocab = build_vocabulary(["good great excellent", "bad terrible awful"] * 20)
    X = encode_texts(
        ["good great excellent", "bad terrible awful"] * 20, vocab, max_len=5
    )
    y = torch.tensor(([1.0, 0.0] * 20))

    model = TextLSTM(vocab_size=len(vocab), embed_dim=16, hidden_dim=8)
    train_classifier(model, X, y, epochs=30, lr=0.01)
    results = evaluate_classifier(model, X, y)

    assert results["accuracy"] > 0.7


def test_pretrained_weights_are_used_when_provided():
    vocab_size = 5
    matrice = torch.zeros(vocab_size, 4)
    matrice[2] = torch.tensor([1.0, 2.0, 3.0, 4.0])

    model = TextGRU(
        vocab_size=vocab_size, embed_dim=4, hidden_dim=3, pretrained_weights=matrice
    )
    assert torch.allclose(model.embedding.weight[2], torch.tensor([1.0, 2.0, 3.0, 4.0]))
