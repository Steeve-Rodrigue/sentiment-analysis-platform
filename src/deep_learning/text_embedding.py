"""
src/deep_learning/text_embedding.py

Phase 5 — Deep Learning — bloc "couche d'embedding et classifieur de référence".

Théorie résumée (voir la conversation associée pour le détail) :

Une couche d'embedding (nn.Embedding) est une simple TABLE DE
CORRESPONDANCE : une ligne par mot du vocabulaire, chaque ligne étant
le vecteur de ce mot. Contrairement à TF-IDF (Phase 2, figé), ces
vecteurs sont des PARAMÈTRES ENTRAÎNABLES, ajustés par rétropropagation
pendant l'entraînement du classifieur -- optimisés pour la tâche de
sentiment, pas pour prédire un contexte.

DEUX FAÇONS D'INITIALISER LA TABLE, vérifiées empiriquement :
1. Aléatoire (from scratch) : le réseau apprend tout depuis zéro.
   Sur 1600 exemples, ~500 000 boutons à régler -> SURAPPRENTISSAGE
   net après quelques dizaines d'époques (train 0.998, test ~0.70).
2. Pré-entraînée (GloVe, Phase 2) : la table démarre avec une vraie
   structure sémantique. Vérifié : 4891/5002 mots du vocabulaire
   trouvés dans GloVe-50 ; après 10 époques, accuracy test = 0.652
   contre 0.590 en aléatoire -- +6 points, car le réseau n'a plus à
   apprendre le sens des mots ET la tâche en même temps.

Le CLASSIFIEUR DE RÉFÉRENCE (baseline) moyenne tous les vecteurs de
mots d'un avis en un seul vecteur, puis applique une couche linéaire.
Volontairement simple : DÉTRUIT l'ordre des mots (même défaut que
TF-IDF) -- sert de point de comparaison pour juger si des architectures
plus complexes (CNN, LSTM, attention -- prochaines notions) apportent
un vrai gain, ou juste de la complexité inutile.

PADDING : les avis n'ont pas tous la même longueur. padding_idx=0
garantit que le vecteur de padding reste à zéro et n'est JAMAIS mis à
jour par le gradient -- vérifié empiriquement.
"""

from __future__ import annotations

from collections import Counter

import torch
import torch.nn as nn


def build_vocabulary(texts: list[str], max_vocab_size: int = 5000) -> dict[str, int]:
    """Construit le vocabulaire à partir d'une liste de textes (train
    UNIQUEMENT, jamais le test -- même règle anti-fuite que TF-IDF en
    Phase 4). id 0 = padding, id 1 = mot inconnu (UNK)."""
    counter = Counter()
    for text in texts:
        counter.update(text.lower().split())

    vocab = {"<pad>": 0, "<unk>": 1}
    for word, _ in counter.most_common(max_vocab_size):
        vocab[word] = len(vocab)
    return vocab


def encode_texts(
    texts: list[str], vocab: dict[str, int], max_len: int = 200
) -> torch.Tensor:
    """Convertit des textes en tenseur d'identifiants, de longueur fixe
    max_len. TRONQUE les textes plus longs (perte d'information au-delà
    de max_len -- compromis pratique pour permettre le traitement par
    lots), COMPLÈTE (padding, id 0) les textes plus courts."""
    encoded = []
    for text in texts:
        ids = [vocab.get(word, 1) for word in text.lower().split()[:max_len]]
        ids += [0] * (max_len - len(ids))
        encoded.append(ids)
    return torch.tensor(encoded)


def build_glove_embedding_matrix(
    vocab: dict[str, int], glove_model, embed_dim: int = 50
) -> torch.Tensor:
    """Construit une matrice d'embedding initialisée avec des vecteurs
    GloVe pré-entraînés (glove_model = objet retourné par
    embeddings.glove_model.load_pretrained_glove(), Phase 2). Les mots
    absents de GloVe gardent un vecteur à zéro (seront appris depuis
    zéro pendant l'entraînement, comme en initialisation aléatoire)."""
    matrix = torch.zeros(len(vocab), embed_dim)
    for word, idx in vocab.items():
        if word in glove_model.key_to_index:
            matrix[idx] = torch.tensor(glove_model[word].copy())
    return matrix


class AverageEmbeddingClassifier(nn.Module):
    """Classifieur de référence : embedding -> moyenne (masquant le
    padding) -> couche linéaire. Détruit volontairement l'ordre des
    mots -- sert de ligne de base pour juger les architectures suivantes."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 50,
        pretrained_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        if pretrained_weights is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_weights, freeze=False, padding_idx=0
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        mask = (x != 0).unsqueeze(-1).float()  # ignore le padding dans la moyenne
        averaged = (embedded * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.fc(averaged).squeeze(-1)


def train_classifier(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int = 10,
    lr: float = 0.001,
    batch_size: int = 32,
) -> dict:
    """Boucle d'entraînement standard pour un classifieur binaire de
    sentiment (les mêmes 4 étapes que dans neural_basics.py)."""
    from torch.utils.data import DataLoader, TensorDataset

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )

    history = []
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
        history.append(float(loss.item()))

    return {"final_loss": history[-1], "history": history}


def evaluate_classifier(
    model: nn.Module, X_test: torch.Tensor, y_test: torch.Tensor
) -> dict:
    """Évalue un classifieur entraîné sur un jeu de test."""
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(X_test))
        predictions = (probabilities > 0.5).float()
        accuracy = (predictions == y_test).float().mean().item()
    return {"accuracy": accuracy, "predictions": predictions}
