"""
src/tokenization/bpe_tokenizer.py

Phase 3 — Tokenisation avancée — bloc "BPE (Byte Pair Encoding)".

Théorie résumée :
BPE part d'un vocabulaire de caractères individuels, puis fusionne
itérativement la paire de symboles adjacents la PLUS FRÉQUENTE dans le
corpus d'entraînement, des milliers de fois. Résultat : les fragments
fréquents de la langue finissent en tokens entiers, les mots rares
restent découpés en plusieurs fragments plus petits -- jamais d'échec
total sur un mot inconnu (contrairement à la tokenisation en mots
entiers de la Phase 1), contrairement aussi à Word2Vec qui levait une
KeyError sur un mot jamais vu.

BPE ne "comprend" rien à la linguistique -- il fusionne uniquement ce
qui est statistiquement fréquent, ce qui le rend applicable à n'importe
quelle langue sans règles écrites à la main (pertinent pour nos 5
langues EN/ES/DE/HI/FR).
"""

from __future__ import annotations

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


def train_bpe_tokenizer(
    corpus: list[str], vocab_size: int = 1000, min_frequency: int = 2
) -> Tokenizer:
    """Entraîne un tokenizer BPE sur un corpus de textes bruts (pas
    besoin de pré-tokeniser en mots au préalable -- Whitespace() s'en
    charge). min_frequency=2 évite de fusionner des paires vues une
    seule fois, ce qui reviendrait à mémoriser le corpus plutôt qu'à
    apprendre des patterns généralisables."""
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=["[UNK]"]
    )
    tokenizer.train_from_iterator(corpus, trainer=trainer)
    return tokenizer


def tokenize_with_bpe(tokenizer: Tokenizer, text: str) -> list[str]:
    """Découpe un texte avec un tokenizer BPE déjà entraîné. Un mot
    jamais vu à l'entraînement est découpé en fragments connus (ou
    caractères isolés / [UNK] en dernier recours) -- jamais un échec
    total, contrairement à un vocabulaire de mots entiers fermé."""
    return tokenizer.encode(text).tokens
