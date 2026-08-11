"""
tests/transformer_arch/test_attention_mechanisms.py

Tests unitaires pour src/transformer_arch/attention_mechanisms.py.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch

from transformers_arch.attention_mechanism import (
    MultiHeadAttention,
    SelfAttention,
    positional_encoding,
    scaled_dot_product_attention,
)


def test_attention_weights_sum_to_one_per_row():
    torch.manual_seed(42)
    q = torch.randn(1, 4, 8)
    k = torch.randn(1, 4, 8)
    v = torch.randn(1, 4, 8)
    _, weights = scaled_dot_product_attention(q, k, v)
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(1, 4), atol=1e-5)


def test_masked_position_gets_exactly_zero_weight():
    q = torch.randn(1, 3, 4)
    k = torch.randn(1, 3, 4)
    v = torch.randn(1, 3, 4)
    mask = torch.zeros(1, 3, 3, dtype=torch.bool)
    mask[0, :, 2] = True  # masque la colonne 2 pour toutes les lignes

    _, weights = scaled_dot_product_attention(q, k, v, mask)
    assert torch.allclose(weights[0, :, 2], torch.zeros(3), atol=1e-6)


def test_self_attention_output_shape_matches_input():
    model = SelfAttention(d_model=8)
    x = torch.randn(2, 5, 8)
    output, weights = model(x)
    assert output.shape == x.shape
    assert weights.shape == (2, 5, 5)


def test_self_attention_is_permutation_dependent_only_with_positions():
    # sans encodage positionnel, le self-attention seul est invariant
    # par permutation -- verifie ici, avant d'ajouter le PE
    torch.manual_seed(42)
    model = SelfAttention(d_model=8)

    mot_a = torch.randn(1, 1, 8)
    mot_b = torch.randn(1, 1, 8)
    mot_c = torch.randn(1, 1, 8)

    phrase_1 = torch.cat([mot_a, mot_b, mot_c], dim=1)
    phrase_2 = torch.cat([mot_c, mot_b, mot_a], dim=1)

    sortie_1, _ = model(phrase_1)
    sortie_2, _ = model(phrase_2)

    assert torch.allclose(sortie_1[0, 0], sortie_2[0, 2], atol=1e-5)


def test_positional_encoding_breaks_permutation_invariance():
    torch.manual_seed(42)
    d_model = 8
    model = SelfAttention(d_model=d_model)

    mot_a = torch.randn(1, 1, d_model)
    mot_b = torch.randn(1, 1, d_model)
    mot_c = torch.randn(1, 1, d_model)

    phrase_1 = torch.cat([mot_a, mot_b, mot_c], dim=1)
    phrase_2 = torch.cat([mot_c, mot_b, mot_a], dim=1)

    pe = positional_encoding(max_len=3, d_model=d_model).unsqueeze(0)

    sortie_1, _ = model(phrase_1 + pe)
    sortie_2, _ = model(phrase_2 + pe)

    assert not torch.allclose(sortie_1[0, 0], sortie_2[0, 2], atol=1e-5)


def test_multi_head_requires_divisible_dimensions():
    try:
        MultiHeadAttention(d_model=10, n_heads=3)
        assert False, "aurait du lever une ValueError"
    except ValueError:
        pass


def test_multi_head_output_shape_matches_input():
    model = MultiHeadAttention(d_model=8, n_heads=2)
    x = torch.randn(2, 5, 8)
    output, weights = model(x)
    assert output.shape == x.shape
    assert weights.shape == (2, 2, 5, 5)  # (batch, n_heads, len, len)


def test_different_heads_produce_different_weights():
    torch.manual_seed(42)
    model = MultiHeadAttention(d_model=8, n_heads=2)
    x = torch.randn(1, 4, 8)
    _, weights = model(x)
    tete_1 = weights[0, 0]
    tete_2 = weights[0, 1]
    assert not torch.allclose(tete_1, tete_2, atol=1e-4)


def test_positional_encoding_gives_unique_vector_per_position():
    pe = positional_encoding(max_len=10, d_model=16)
    for i in range(10):
        for j in range(i + 1, 10):
            assert not torch.allclose(pe[i], pe[j], atol=1e-6)
