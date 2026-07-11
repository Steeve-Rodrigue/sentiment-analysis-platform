"""
src/preprocessing/cleaning.py

Phase 1 — NLP Fundamentals — bloc "nettoyage de texte bruité".

Couvre : normalisation Unicode, expressions régulières, suppression d'URL,
traitement des hashtags, gestion des emojis, réduction des répétitions.

Chaque fonction a été construite et testée pas à pas (voir la conversation
associée) avant d'être assemblée ici dans clean_text(). L'ordre des étapes
dans clean_text() est volontaire :
1. Unicode d'abord (sinon les regex suivantes peuvent rater des caractères
   mal encodés).
2. URLs et mentions avant tout nettoyage générique (sinon leurs '/' et '.'
   seraient déjà charcutés par une autre étape).
3. Hashtags et emojis ensuite, une fois le texte débarrassé du bruit structurel.
4. Réduction des répétitions et mise en minuscule en dernier (la casse doit
   rester intacte pendant la séparation CamelCase des hashtags).
"""

from __future__ import annotations

import re
import unicodedata

import emoji

# --- Motifs regex compilés une seule fois (réutilisés à chaque appel) ---

URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
MENTION_PATTERN = re.compile(r"@(\w+)")
HASHTAG_PATTERN = re.compile(r"#(\w+)")
REPEATED_CHARS_PATTERN = re.compile(r"(.)\1{2,}")
REPEATED_PUNCT_PATTERN = re.compile(r"([!?.]){2,}")
WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_unicode(text: str, form: str = "NFKC") -> str:
    """Canonicalise l'encodage Unicode. NFKC unifie aussi bien les formes
    composées/décomposées (é vs e + accent) que les variantes de
    compatibilité (chiffres pleine largeur, ligatures) — plus sûr que NFC
    seul pour du texte utilisateur bruité."""
    return unicodedata.normalize(form, text)


def remove_urls(text: str, placeholder: str = "") -> str:
    """Retire les URLs (avec ou sans protocole http/https)."""
    return URL_PATTERN.sub(placeholder, text)


def remove_mentions(text: str, placeholder: str = "") -> str:
    """Retire les mentions de type @pseudo."""
    return MENTION_PATTERN.sub(placeholder, text)


def process_hashtags(text: str, mode: str = "split_camel") -> str:
    """Traite les hashtags. mode="remove" les supprime entièrement (perte
    de signal) ; mode="unwrap" garde le mot collé (#GreatService ->
    GreatService) ; mode="split_camel" (par défaut) sépare aussi le
    CamelCase interne (#GreatService -> Great Service), le plus exploitable
    pour la tokenisation qui suit."""
    if mode == "remove":
        return HASHTAG_PATTERN.sub("", text)
    if mode == "split_camel":
        def _split(m: re.Match) -> str:
            word = m.group(1)
            return re.sub(r"(?<!^)(?=[A-Z])", " ", word)
        return HASHTAG_PATTERN.sub(_split, text)
    return HASHTAG_PATTERN.sub(r"\1", text)  # mode "unwrap"


def demojize(text: str) -> str:
    """Convertit les emojis en tokens texte (ex. '😡' -> '__EMOJI_enraged_face__')
    plutôt que de les supprimer, pour préserver leur signal de sentiment
    souvent plus fiable que le texte environnant."""
    return emoji.demojize(text, delimiters=(" __EMOJI_", "__ "))


def remove_emoji(text: str) -> str:
    """Alternative : supprime les emojis entièrement (perd le signal)."""
    return emoji.replace_emoji(text, replace="")


def reduce_repeated_chars(text: str, max_repeat: int = 2) -> str:
    """"sooooo" -> "soo" : garde l'emphase sans exploser le vocabulaire
    avec chaque variante d'élongation."""
    return REPEATED_CHARS_PATTERN.sub(lambda m: m.group(1) * max_repeat, text)


def normalize_repeated_punctuation(text: str) -> str:
    """"!!!!" -> "!" """
    return REPEATED_PUNCT_PATTERN.sub(r"\1", text)


def normalize_whitespace(text: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def clean_text(text: str, lowercase: bool = True) -> str:
    """Pipeline de nettoyage complet, dans l'ordre justifié en en-tête de
    ce fichier. Chaque étape a été testée individuellement avant d'être
    assemblée ici."""
    text = normalize_unicode(text)
    text = remove_urls(text)
    text = remove_mentions(text)
    text = process_hashtags(text)
    text = demojize(text)
    text = reduce_repeated_chars(text)
    text = normalize_repeated_punctuation(text)
    if lowercase:
        text = text.lower()
    text = normalize_whitespace(text)
    return text