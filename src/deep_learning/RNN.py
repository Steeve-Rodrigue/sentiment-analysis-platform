"""
src/deep_learning/rnn_models.py

Phase 5 — Deep Learning — bloc "RNN, LSTM, Bi-LSTM, GRU".

Théorie résumée (voir la conversation associée pour le détail) :

Un RNN simple lit la phrase mot par mot, maintenant un état caché h_t
mis à jour à chaque étape : h_t = f(h_{t-1}, x_t). Problème vérifié
empiriquement : sur une séquence de 15 mots, changer RADICALEMENT le
tout premier mot produit une différence de 0.000000 sur l'état final --
le signal est totalement effacé (la tanh répétée sature la mémoire).

LSTM corrige ça avec un état de cellule séparé (c_t), protégé par 3
portes apprises (sigmoïde, valeurs entre 0 et 1) :
  - porte d'OUBLI  : combien de la mémoire précédente garder
  - porte d'ENTRÉE : combien du nouveau candidat ajouter
  - porte de SORTIE: combien de la mémoire exposer comme sortie visible
Vérifié : sur la même séquence de 15 mots, le LSTM garde une trace
MESURABLE (non nulle) du premier mot, contrairement au RNN simple.
Vérifié aussi : une réimplémentation manuelle des 4 portes reproduit
EXACTEMENT (torch.allclose) la sortie de nn.LSTMCell -- confirme la
compréhension du mécanisme, pas juste une boîte noire.

Bi-LSTM (bidirectionnel) : lit la phrase dans les DEUX sens (gauche à
droite ET droite à gauche), puis concatène les deux états finaux. Utile
car le sens d'un mot dépend parfois du contexte à VENIR, pas seulement
de ce qui précède ("the film was, despite early doubts, excellent" --
comprendre "doubts" comme surmonté demande de voir "excellent" après).

GRU (Gated Recurrent Unit) : simplification du LSTM à 2 portes au lieu
de 3 (fusionne oubli/entrée en une porte "update", ajoute une porte
"reset"). Moins de paramètres, entraînement plus rapide, performance
souvent comparable au LSTM en pratique -- un compromis vitesse/capacité.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TextLSTM(nn.Module):
    """LSTM unidirectionnel : ne lit la phrase que de gauche à droite."""

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
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        _, (last_hidden, _) = self.lstm(embedded)
        return self.fc(last_hidden.squeeze(0)).squeeze(-1)


class TextBiLSTM(nn.Module):
    """LSTM bidirectionnel : lit la phrase dans les deux sens, concatène
    les deux états finaux -- capture le contexte AVANT et APRÈS chaque mot."""

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
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        # bidirectional=True double la dimension de sortie (avant + arriere concatenes)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        _, (last_hidden, _) = self.lstm(embedded)
        # last_hidden : (2, batch, hidden_dim) -- direction avant et arriere separement
        concatenated = torch.cat([last_hidden[0], last_hidden[1]], dim=1)
        return self.fc(concatenated).squeeze(-1)


class TextGRU(nn.Module):
    """GRU : simplification du LSTM (2 portes au lieu de 3), moins de
    paramètres, entraînement plus rapide."""

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
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        _, last_hidden = self.gru(
            embedded
        )  # GRU n'a pas d'etat de cellule separe (pas de tuple)
        return self.fc(last_hidden.squeeze(0)).squeeze(-1)
