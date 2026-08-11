"""
tests/transformer_arch/test_encoder.py

Tests unitaires pour src/transformer_arch/encoder.py.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch
from torch import nn

from transformers_arch.encoder import EncoderBlock, TransformerEncoder


def test_encoder_block_output_shape_matches_input():
    block = EncoderBlock(d_model=8, n_heads=2, d_ff=32)
    x = torch.randn(2, 5, 8)
    output, weights = block(x)
    assert output.shape == x.shape


def test_stacked_encoder_output_shape_matches_input():
    encoder = TransformerEncoder(d_model=8, n_heads=2, d_ff=32, n_layers=3)
    x = torch.randn(2, 5, 8)
    output = encoder(x)
    assert output.shape == x.shape


def test_residual_connections_preserve_gradient_through_depth():
    # regression du phenomene verifie dans la conversation : sans
    # residuel, le gradient meurt sur un empilement profond
    torch.manual_seed(42)

    class BlocSansResiduel(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.couche = nn.Linear(dim, dim)

        def forward(self, x):
            return torch.relu(self.couche(x))

    class BlocAvecResiduel(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.couche = nn.Linear(dim, dim)

        def forward(self, x):
            return x + torch.relu(self.couche(x))

    dim, depth = 16, 30
    reseau_sans = nn.Sequential(*[BlocSansResiduel(dim) for _ in range(depth)])
    reseau_avec = nn.Sequential(*[BlocAvecResiduel(dim) for _ in range(depth)])

    x1 = torch.randn(1, dim, requires_grad=True)
    reseau_sans(x1).sum().backward()
    grad_sans = x1.grad.abs().mean().item()

    x2 = x1.detach().clone().requires_grad_(True)
    reseau_avec(x2).sum().backward()
    grad_avec = x2.grad.abs().mean().item()

    assert grad_sans < 1e-6
    assert grad_avec > 1.0


def test_encoder_block_output_is_normalized():
    # LayerNorm garantit une moyenne proche de 0 par position
    torch.manual_seed(42)
    block = EncoderBlock(d_model=16, n_heads=4, d_ff=64)
    x = torch.randn(1, 3, 16) * 100  # entree volontairement mal calibree
    output, _ = block(x)
    moyenne_par_position = output.mean(dim=-1)
    assert torch.allclose(moyenne_par_position, torch.zeros(1, 3), atol=1e-4)


def test_multi_layer_encoder_accepts_mask():
    encoder = TransformerEncoder(d_model=8, n_heads=2, d_ff=16, n_layers=2)
    x = torch.randn(1, 4, 8)
    mask = torch.zeros(1, 4, 4, dtype=torch.bool)
    mask[0, :, 3] = True  # masque la derniere position (ex: padding)
    output = encoder(x, mask)
    assert output.shape == x.shape
