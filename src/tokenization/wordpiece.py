"""
src/tokenization/wordpiece_tokenizer.py

Phase 3 — Tokenisation avancée — bloc "WordPiece".

Théorie résumée  :
WordPiece (utilisé par BERT) diffère de BPE sur UN point précis : au
lieu de fusionner la paire la PLUS FRÉQUENTE (BPE), il fusionne la paire
au score de VRAISEMBLANCE le plus élevé :

    score(a, b) = freq(a, b) / (freq(a) * freq(b))

Ce score normalise la fréquence brute par la fréquence individuelle de
chaque symbole -- une paire où les deux symboles sont rares
individuellement mais TOUJOURS ensemble obtient un score plus élevé
qu'une paire de symboles très fréquents qui se croisent en partie par
hasard. Vérifié empiriquement sur un corpus jouet : BPE choisit sa
première fusion sur la fréquence brute (13 occurrences), WordPiece
choisit une paire bien moins fréquente en absolu (6 occurrences) mais
mieux "corrélée" statistiquement.

Convention de notation : les fragments qui continuent un mot (pas le
tout premier) sont préfixés par "##" -- ex. "delivery" -> ["del", "##i",
"##v", "##er", "##y"] -- pour indiquer explicitement qu'ils ne
commencent pas un nouveau mot, contrairement à BPE qui n'a pas cette
convention.
"""

from __future__ import annotations

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer


def train_wordpiece_tokenizer(
    corpus: list[str], vocab_size: int = 1000, min_frequency: int = 2
) -> Tokenizer:
    """Entraîne un tokenizer WordPiece sur un corpus de textes bruts.
    Structure identique à train_bpe_tokenizer() -- seul le modèle
    interne (score de vraisemblance vs fréquence brute) diffère."""
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordPieceTrainer(
        vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=["[UNK]"]
    )
    tokenizer.train_from_iterator(corpus, trainer=trainer)
    return tokenizer


def tokenize_with_wordpiece(tokenizer: Tokenizer, text: str) -> list[str]:
    """Découpe un texte avec un tokenizer WordPiece déjà entraîné. Les
    fragments continuant un mot portent le préfixe "##"."""
    return tokenizer.encode(text).tokens
