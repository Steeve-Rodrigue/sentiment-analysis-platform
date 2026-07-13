"""
src/preprocessing/tokenization.py

Phase 1 — NLP Fundamentals — bloc "tokenisation".

Couvre : tokenisation par espaces (whitespace), tokenisation par règles
linguistiques (mots), tokenisation de phrases, et leur combinaison.

Chaque fonction a été construite, comparée et testée pas à pas (voir la
conversation associée) avant d'être regroupée ici.

Note volontaire : la tokenisation en sous-mots (BPE, WordPiece,
SentencePiece) n'est PAS traitée ici — elle fait l'objet de la Phase 3,
qui mérite sa propre théorie (compression, gestion des mots hors
vocabulaire).
"""

from __future__ import annotations

from nltk.tokenize import sent_tokenize, word_tokenize, wordpunct_tokenize


def whitespace_tokenize(text: str) -> list[str]:
    """Le tokenizer le plus simple possible : découpe uniquement sur les
    espaces. Rapide, mais naïf : "great!" reste un seul token, distinct
    de "great" — ce qui double le vocabulaire pour rien."""
    return text.split()


def rule_based_tokenize(text: str, language: str = "english") -> list[str]:
    """Tokenizer de type Treebank (NLTK) : sépare la ponctuation des mots
    et gère les contractions ("don't" -> "do" + "n't") via des règles
    linguistiques, pas juste des classes de caractères."""
    return word_tokenize(text, language=language)


def punctuation_split_tokenize(text: str) -> list[str]:
    """Découpe aussi finement la ponctuation ET les espaces comme classes
    de tokens séparées (ex. "don't" -> ["don", "'", "t"]). Utile en
    solution de repli pour des langues/scripts où les règles spécifiques
    de NLTK ne s'appliquent pas bien."""
    return wordpunct_tokenize(text)


def sentence_tokenize(text: str, language: str = "english") -> list[str]:
    """Découpe un texte en phrases en gérant correctement les abréviations
    (Punkt est un modèle statistique pré-entraîné, pas une regex naïve sur
    le point '.')."""
    return sent_tokenize(text, language=language)


def tokenize_document(text: str, language: str = "english") -> list[list[str]]:
    """Tokenisation à deux niveaux : phrases, puis mots à l'intérieur de
    chaque phrase. C'est la structure attendue par la plupart des étapes
    suivantes (POS tagging, dependency parsing), qui opèrent au niveau de
    la phrase, pas du document entier."""
    return [
        rule_based_tokenize(sent, language=language)
        for sent in sentence_tokenize(text, language=language)
    ]
