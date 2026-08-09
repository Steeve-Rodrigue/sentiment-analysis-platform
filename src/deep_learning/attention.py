"""
src/deep_learning/attention.py

Phase 5 -- Deep Learning -- bloc "attention".

Theorie resumee (voir la conversation associee pour le detail) :

Le LSTM classique (rnn_models.TextLSTM) ne garde que le DERNIER etat
cache pour decider -- un goulot d'etranglement : toute l'information
de la phrase doit passer par ce seul vecteur, meme avec les portes du
LSTM qui protegent mieux la memoire que le RNN simple.

L'attention regarde TOUS les etats caches (un par mot), calcule un
score d'importance pour chacun, puis combine tout en une somme
PONDEREE plutot qu'un simple dernier etat :

    contexte = somme_i( poids_i * etat_i )   avec  somme_i(poids_i) = 1

Les poids sont obtenus via softmax sur les scores bruts -- garantit
qu'ils somment a 1 et restent positifs, lisibles comme des
probabilites d'importance.

Verifie empiriquement sur un pattern clair ("the delivery was
terrible" vs "...excellent") : apres entrainement, le mot porteur de
sentiment ("terrible") recoit 0.731 de poids d'attention, contre
0.04-0.12 pour les mots grammaticaux ("the", "was") -- le modele
identifie seul, sans supervision explicite sur quel mot compte, le mot
qui porte le sentiment.

PIEGE TECHNIQUE a connaitre : sans masquage du padding, les positions
de padding recoivent quand meme un score et polluent la softmax avec
du bruit. masked_fill(mask == 0, -inf) AVANT la softmax force leur
poids a exactement 0 (exp(-inf) = 0) -- verifie empiriquement.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMWithAttention(nn.Module):
    """LSTM + couche d'attention, pour la classification de sentiment.

    Contrairement a TextLSTM (rnn_models.py), qui n'utilise que le
    dernier etat cache, ce modele combine TOUS les etats via une
    somme ponderee -- retourne aussi les poids d'attention si demande,
    utile pour l'interpretabilite (lien direct avec la Phase 12,
    IA explicable)."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 50,
        hidden_dim: int = 32,
        pretrained_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        if pretrained_weights is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_weights, freeze=False, padding_idx=0
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, return_weights: bool = False) -> torch.Tensor:
        mask = (x != 0).float()
        embedded = self.embedding(x)
        hidden_states, _ = self.lstm(embedded)  # tous les etats, pas 1 seul

        scores = self.attention(hidden_states).squeeze(-1)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=1)

        context = (hidden_states * weights.unsqueeze(-1)).sum(dim=1)
        output = self.fc(context).squeeze(-1)

        if return_weights:
            return output, weights
        return output


def get_attention_weights(model: LSTMWithAttention, x: torch.Tensor) -> torch.Tensor:
    """Recupere uniquement les poids d'attention pour un batch donne,
    sans recalculer manuellement le forward pass complet."""
    model.eval()
    with torch.no_grad():
        _, weights = model(x, return_weights=True)
    return weights
