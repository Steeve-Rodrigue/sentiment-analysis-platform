"""
src/transformer_arch/encoder.py

Phase 6 -- Transformers -- bloc "Encodeur (residuel, feed-forward,
layer norm)".

Un bloc encodeur combine le MultiHeadAttention deja construit avec
deux ajouts :
1. Feed-Forward Network -- 2 couches Linear + activation, appliquees
   INDEPENDAMMENT a chaque position, qui "digerent" ce que l'attention
   vient de rassembler.
2. Connexions residuelles (x + Sublayer(x)) + LayerNorm apres chaque
   sous-couche.

POURQUOI LES CONNEXIONS RESIDUELLES SONT INDISPENSABLES, verifie
empiriquement sur une pile de 30 blocs sans/avec residuel :
    SANS residuel : gradient a l'entree = 0.00000000 (mort)
    AVEC residuel : gradient a l'entree = 583.95 (vivant)
Meme phenomene que le "gradient qui s'evanouit" observe avec le RNN
simple en Phase 5, mais ici cause par la PROFONDEUR du reseau (couches
empilees), pas la longueur de la sequence. Le raccourci "+ x" permet
au gradient de revenir en arriere sans etre force de traverser toute
la transformation.

LayerNorm recentre/redimensionne apres chaque sous-couche (moyenne 0,
ecart-type 1, par position) -- stabilise l'entrainement en evitant que
les valeurs n'explosent ou ne s'effondrent au fil des couches
empilees.

La forme de sortie du bloc est IDENTIQUE a la forme d'entree -- c'est
ce qui permet d'empiler plusieurs blocs a la suite (BERT en empile 12,
GPT-3 en empile 96), chaque bloc affinant un peu plus la
representation.
"""

from __future__ import annotations

import torch
from torch import nn

from transformers_arch.attention_mechanism import MultiHeadAttention


class EncoderBlock(nn.Module):
    """Un bloc encodeur de Transformer : multi-head attention +
    residuel + norm, puis feed-forward + residuel + norm."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, weights = self.attention(x, mask)
        x = self.norm1(x + attn_out)
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        return x, weights


class TransformerEncoder(nn.Module):
    """Empile plusieurs EncoderBlock a la suite -- la forme de sortie
    de chaque bloc etant identique a sa forme d'entree, ils peuvent
    s'enchainer directement."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        for layer in self.layers:
            x, _ = layer(x, mask)
        return x
