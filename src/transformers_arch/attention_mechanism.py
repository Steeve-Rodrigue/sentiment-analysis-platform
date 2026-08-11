"""
src/transformer_arch/attention_mechanism.py

Phase 6 -- Transformers -- bloc "self-attention, multi-head, positional
encoding".

SELF-ATTENTION : contrairement a l'attention de la Phase 5 (un seul
score par mot, par rapport a une direction unique apprise), le
self-attention calcule un score de pertinence ENTRE CHAQUE PAIRE de
mots. Chaque mot est projete en 3 versions (Query, Key, Value) via des
couches Linear ordinaires -- memes poids, meme cycle d'apprentissage
(retropropagation) que partout ailleurs dans ce projet.

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

Verifie empiriquement : chaque LIGNE de la matrice de poids somme a 1
(softmax appliquee dim=-1) -- une ligne = un mot qui distribue son
attention, une colonne = un mot qui la recoit.

MULTI-HEAD : plusieurs "tetes" d'attention en parallele, chacune sur
une portion plus petite de la dimension totale (d_k = d_model /
n_heads). Verifie : deux tetes independantes produisent des matrices
de poids DIFFERENTES -- chacune peut se specialiser sur un type de
relation different.

POSITIONAL ENCODING : le self-attention seul est INVARIANT PAR
PERMUTATION -- verifie empiriquement (torch.allclose == True) que le
vecteur de sortie du mot "a" est identique qu'il soit en position 0 ou
en position 2 d'une phrase. Le positional encoding (fonctions sin/cos
a differentes frequences, ajoutees a l'embedding) casse cette symetrie
-- verifie : apres ajout du PE, la meme comparaison donne
torch.allclose == False.

MASQUAGE (-inf) : pour interdire certaines positions (padding, ou plus
tard les mots futurs dans un decodeur), on met le score BRUT a -inf
AVANT la softmax, pas le poids final a 0 apres coup -- exp(-inf) = 0
exactement, et les autres poids de la meme ligne se renormalisent
automatiquement pour sommer a 1 entre eux.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calcule Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V.

    mask (optionnel) : booleen, True = position a MASQUER (mise a
    -inf avant softmax). Retourne (sortie, poids_attention)."""
    d_k = query.size(-1)
    scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    output = weights @ value
    return output, weights


class SelfAttention(nn.Module):
    """Une seule tete de self-attention -- projette l'entree en
    Query/Key/Value via 3 couches Linear independantes, puis applique
    scaled_dot_product_attention."""

    def __init__(self, d_model: int):
        super().__init__()
        self.query_proj = nn.Linear(d_model, d_model, bias=False)
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.value_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.query_proj(x)
        key = self.key_proj(x)
        value = self.value_proj(x)
        return scaled_dot_product_attention(query, key, value, mask)


class MultiHeadAttention(nn.Module):
    """Plusieurs tetes de self-attention en parallele, chacune sur une
    portion de d_model, recombinees par une projection finale W_o."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model doit etre divisible par n_heads")
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.query_proj = nn.Linear(d_model, d_model, bias=False)
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.value_proj = nn.Linear(d_model, d_model, bias=False)
        self.output_proj = nn.Linear(d_model, d_model, bias=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        x = x.view(batch, length, self.n_heads, self.d_k)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, d_model = x.shape

        query = self._split_heads(self.query_proj(x))
        key = self._split_heads(self.key_proj(x))
        value = self._split_heads(self.value_proj(x))

        output, weights = scaled_dot_product_attention(query, key, value, mask)

        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(batch, length, d_model)
        return self.output_proj(output), weights


def positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """Construit la matrice d'encodage positionnel sinusoidal
    (max_len x d_model). Formule Vaswani et al. 2017 :
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))"""
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
