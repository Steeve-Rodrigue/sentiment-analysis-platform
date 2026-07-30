"""
src/tokenization/byte_level_bpe.py

Phase 3 — Tokenisation avancée — bloc "Byte-level BPE" (approche GPT-2).

Le BPE classique (bpe_tokenizer.py) opère sur des CARACTÈRES Unicode --
il existe des centaines de milliers de caractères Unicode possibles,
impossible de garantir qu'on les a tous vus à l'entraînement (d'où les
tokens [UNK] observés sur des caractères jamais rencontrés).

Byte-level BPE opère sur les OCTETS bruts de l'encodage UTF-8 : un octet
ne prend que 256 valeurs possibles, un ensemble FIXE et FINI connu à
l'avance. En forçant ces 256 valeurs dans le vocabulaire dès le départ
(initial_alphabet=ByteLevel.alphabet()), on garantit mathématiquement
qu'AUCUN texte, dans AUCUNE langue, avec AUCUN emoji, ne peut jamais
produire de token [UNK] -- vérifié empiriquement sur emoji et japonais.

PIÈGE RENCONTRÉ ET CORRIGÉ pendant l'apprentissage : sans
initial_alphabet=ByteLevel.alphabet(), le trainer n'ajoute que les
octets VUS dans le corpus d'entraînement au vocabulaire -- la garantie
"jamais d'OOV" ne tient alors plus du tout (des caractères jamais vus
disparaissaient silencieusement au lieu d'être décomposés en octets).
Ce paramètre n'est PAS optionnel si on veut la propriété qu'on recherche.
"""

from __future__ import annotations

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


def train_byte_level_bpe(
    corpus: list[str], vocab_size: int = 1000, min_frequency: int = 2
) -> Tokenizer:
    """Entraîne un tokenizer Byte-level BPE (façon GPT-2). Le paramètre
    initial_alphabet est ce qui garantit la couverture totale des 256
    octets possibles dès le départ -- sans lui, la garantie "jamais
    d'OOV" ne tient pas (voir piège documenté en en-tête de fichier)."""
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        initial_alphabet=ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(corpus, trainer=trainer)
    return tokenizer


def tokenize_with_byte_level_bpe(tokenizer: Tokenizer, text: str) -> list[str]:
    """Découpe un texte en tokens de niveau octet."""
    return tokenizer.encode(text).tokens


def detokenize_with_byte_level_bpe(tokenizer: Tokenizer, tokens: list[str]) -> str:
    """Reconstruit le texte original à partir des tokens -- garanti
    exact quel que soit le texte d'origine (contrairement à BPE
    caractère-niveau, qui peut perdre des caractères jamais vus)."""
    ids = [tokenizer.token_to_id(tok) for tok in tokens]
    return tokenizer.decode(ids)
