"""
src/deep_learning/text_cnn.py

Phase 5 — Deep Learning — bloc "CNN pour texte".

Théorie résumée (voir la conversation associée pour le détail) :
Le classifieur de référence (text_embedding.AverageEmbeddingClassifier)
moyenne tous les embeddings d'un avis -- une opération COMMUTATIVE,
donc totalement aveugle à l'ordre des mots. Vérifié empiriquement :
"not very good" et "good very not" (mêmes mots, ordre inversé)
produisent EXACTEMENT le même vecteur moyen (torch.allclose == True).

Un CNN corrige ça avec des filtres convolutionnels : une fenêtre
glissante de N mots consécutifs (kernel_size) détecte des motifs
LOCAUX, dans leur ORDRE exact. Vérifié empiriquement : les deux mêmes
phrases inversées donnent cette fois des sorties CNN différentes
(torch.allclose == False) -- la différence se joue précisément au
niveau de l'AGRÉGATION (moyenne globale vs fenêtres locales ordonnées),
pas de l'embedding lui-même (identique dans les deux architectures).

Après la convolution, le MAX-POOLING garde le meilleur score de chaque
filtre sur toute la phrase ("peu importe où le motif apparaît, s'il
apparaît, ça compte"), ramenant une phrase de longueur variable à un
vecteur de taille FIXE.

kernel_sizes=(3,4,5) : plusieurs longueurs de motifs recherchées en
parallèle (trigrammes, 4-grammes, 5-grammes) -- chaque longueur explore
un nombre différent de positions dans la phrase (vérifié : 8 mots ->
6 positions en taille 3, 5 en taille 4, 4 en taille 5), mais après
pooling chacune se réduit à un score fixe par filtre.

num_filters : combien de motifs DIFFÉRENTS chercher, pour CHAQUE
longueur -- chaque filtre peut en théorie se spécialiser sur un motif
distinct ("not very good", "absolutely amazing"...).

Les scores de tous les filtres, toutes tailles confondues, sont
CONCATÉNÉS avant la couche de classification finale.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TextCNN(nn.Module):
    """CNN pour classification de texte. Réutilise le padding_idx=0
    de la couche d'embedding (même garantie que dans
    AverageEmbeddingClassifier : le vecteur de padding reste à zéro)."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 50,
        num_filters: int = 32,
        kernel_sizes: tuple[int, ...] = (3, 4, 5),
        pretrained_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        if pretrained_weights is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_weights, freeze=False, padding_idx=0
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, num_filters, kernel_size=k) for k in kernel_sizes]
        )
        self.fc = nn.Linear(num_filters * len(kernel_sizes), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x).permute(
            0, 2, 1
        )  # (batch, embed_dim, seq_len) -- format attendu par Conv1d
        pooled = [torch.relu(conv(embedded)).max(dim=2).values for conv in self.convs]
        concatenated = torch.cat(pooled, dim=1)
        return self.fc(concatenated).squeeze(-1)
