"""
src/tokenization/sentencepiece_tokenizer.py

Phase 3 — Tokenisation avancée — bloc "SentencePiece".

Contrairement à BPE/WordPiece (qui dépendent d'un pré-tokenizer par
espaces, une hypothèse fausse pour le japonais/chinois par exemple),
SentencePiece traite le texte comme un flux BRUT de caractères Unicode,
sans aucune étape de pré-découpage. L'espace lui-même devient un
caractère appris comme un autre, représenté par le symbole "▁"
(U+2581). Conséquence : le pipeline est réversible -- encode() puis
decode() peut reconstruire le texte original exactement, y compris son
espacement -- ce que BPE/WordPiece avec pré-tokenizer par espaces ne
garantissent pas.

Vérifié empiriquement : le MÊME code, sans aucune modification, traite
correctement l'anglais, le français ET le hindi (script devanagari) --
c'est pourquoi les modèles multilingues comme XLM-R (Phase 8) s'appuient
sur SentencePiece plutôt que sur BPE/WordPiece classiques.

Différence d'API à connaître : SentencePiece attend un FICHIER sur
disque en entrée (pas une liste Python en mémoire comme `tokenizers`).
"""

from __future__ import annotations

import sentencepiece as spm


def train_sentencepiece_tokenizer(
    corpus_file_path: str,
    model_prefix: str,
    vocab_size: int = 1000,
    model_type: str = "bpe",
    character_coverage: float = 1.0,
) -> spm.SentencePieceProcessor:
    """Entraîne un tokenizer SentencePiece depuis un fichier texte brut
    (un exemple par ligne). character_coverage=1.0 garantit une bonne
    couverture des scripts non-latins (devanagari pour le hindi,
    notamment) -- une valeur plus basse risquerait de traiter des
    caractères rares comme hors-vocabulaire."""
    spm.SentencePieceTrainer.train(
        input=corpus_file_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
    )
    return spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")


def tokenize_with_sentencepiece(
    processor: spm.SentencePieceProcessor, text: str
) -> list[str]:
    """Découpe un texte en tokens SentencePiece (le symbole ▁ marque un
    espace précédant le token)."""
    return processor.encode(text, out_type=str)


def detokenize_with_sentencepiece(
    processor: spm.SentencePieceProcessor, tokens: list[str]
) -> str:
    """Reconstruit le texte original à partir des tokens -- opération
    fiable grâce à l'encodage explicite des espaces (▁), contrairement à
    un simple `" ".join(tokens)` qui perdrait l'espacement d'origine."""
    return processor.decode(tokens)
