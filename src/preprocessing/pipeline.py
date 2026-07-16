"""
src/preprocessing/pipeline.py

Phase 1 — NLP Fundamentals — assemblage final.

Combine les 7 modules de la Phase 1 en une seule fonction preprocess(),
dans l'ordre justifié tout au long de la phase  :

1. detect_language()      -- décide quelles ressources linguistiques utiliser
2. clean_text()            -- Unicode, URLs, hashtags, emojis, répétitions
3. rule_based_tokenize()    -- découpe en mots
4. correct_spelling()       -- tôt, avant que les étapes suivantes aient besoin
                               de mots correctement orthographiés
5. mark_negation_scope()    -- AVANT remove_stopwords(), sinon "not" disparaît
                               et "good_NEG" ne peut jamais se former
6. remove_stopwords()       -- optionnel : utile pour Phase 4 (Bag-of-Words),
                               à éviter pour les Transformers (Phase 6+)
7. lemmatize_tokens()       -- dernière étape, réduit à la forme dictionnaire

POS tagging, NER et dependency parsing (linguistic.py) ne sont PAS inclus
dans cette chaîne linéaire -- ce sont des outils utilisés ponctuellement
(ex. extraction d'aspects candidats en Phase 9), pas une étape systématique
de nettoyage/normalisation.
"""

from __future__ import annotations

from .cleaning import clean_text
from .language_detection import detect_language
from .negation import mark_negation_scope
from .normalization import lemmatize_tokens, remove_stopwords
from .spelling import correct_spelling
from .tokenization import rule_based_tokenize

# NLTK attend des noms de langue complets ("english"), pas des codes
# ISO-639-1 ("en") -- cette table fait le lien entre les deux mondes.
# Le hindi n'a pas de liste de stop-words NLTK ; on le gère explicitement
# plus bas plutôt que de laisser un KeyError silencieux se produire.
_ISO_TO_NLTK_LANGUAGE = {
    "en": "english",
    "es": "spanish",
    "de": "german",
    "fr": "french",
}


def preprocess(
    text: str,
    language: str | None = None,
    apply_spelling: bool = True,
    apply_negation: bool = True,
    apply_stopwords: bool = True,
    apply_lemmatization: bool = True,
) -> dict:
    """Pipeline complet de la Phase 1. Retourne un dict avec la langue
    détectée (ou fournie) et les tokens finaux, prêts pour la Phase 2.

    Chaque étape est optionnelle via un paramètre, car le bon réglage
    dépend du modèle en aval (ex. apply_stopwords=False pour un
    Transformer en Phase 6, True pour un modèle Bag-of-Words en Phase 4).
    """
    if language is None:
        language = detect_language(text)

    nltk_language = _ISO_TO_NLTK_LANGUAGE.get(language, "english")

    cleaned = clean_text(text)
    tokens = rule_based_tokenize(cleaned, language=nltk_language)

    if apply_spelling:
        tokens = correct_spelling(tokens)

    if apply_negation:
        tokens = mark_negation_scope(tokens)

    if apply_stopwords and language in _ISO_TO_NLTK_LANGUAGE:
        # pas de liste de stop-words NLTK pour le hindi -- on saute
        # plutôt que de planter, en le documentant explicitement ici
        tokens = remove_stopwords(tokens, language=nltk_language)

    if apply_lemmatization:
        tokens = lemmatize_tokens(tokens)

    return {
        "language": language,
        "tokens": tokens,
    }
