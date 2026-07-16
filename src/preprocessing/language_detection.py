"""
src/preprocessing/language_detection.py

Phase 1 — NLP Fundamentals — bloc "détection de langue".
Dernière notion de la Phase 1 — c'est le point d'entrée du pipeline
multilingue : avant toute autre étape (nettoyage, tokenisation,
normalisation), il faut savoir dans quelle langue est un avis pour
appliquer les bonnes ressources (stopwords, stemmer, modèle spaCy).

Théorie résumée (voir la conversation associée pour le détail) :
langdetect compare le profil statistique de n-grammes de caractères d'un
texte à des profils appris par langue (façon Naive Bayes, sur Wikipedia).
Limite IMPORTANTE, vérifiée empiriquement : sur des textes courts et
ambigus ("No", "Si", "Bien" -- des mots valides dans plusieurs langues),
l'algorithme ne se contente pas d'être incertain, il est CONFIANT à
>99.9% tout en se trompant. Un avis client très court ("Great!", "Nul.")
est exactement ce type de cas à risque -- à garder en tête pour la
Phase 13 (human-in-the-loop sur les détections peu fiables).
"""

from __future__ import annotations

from langdetect import DetectorFactory, detect, detect_langs

# Rend les résultats reproductibles -- l'algorithme de langdetect a un
# élément aléatoire dans son échantillonnage de n-grammes ; sans fixer
# cette graine, un même texte pourrait occasionnellement changer de
# langue détectée d'un appel à l'autre.
DetectorFactory.seed = 0

SUPPORTED_LANGUAGES = {"en", "es", "de", "hi", "fr"}


def detect_language(text: str) -> str:
    """Retourne le code ISO-639-1 de la langue la plus probable.
    Retourne "unknown" si le texte est trop court/ambigu pour être
    classé (langdetect lève une exception dans ce cas)."""
    try:
        return detect(text)
    except Exception:
        return "unknown"


def detect_language_probabilities(text: str) -> list[tuple[str, float]]:
    """Retourne [(code_langue, probabilité), ...] triés par probabilité.
    Wrapper autour de langdetect.detect_langs() : convertit ses objets
    Language en simples tuples (str, float), et ne plante pas sur texte
    vide/non détectable (renvoie [] au lieu de lever une exception)."""
    try:
        return [(r.lang, r.prob) for r in detect_langs(text)]
    except Exception:
        return []


def is_supported_language(text: str) -> bool:
    return detect_language(text) in SUPPORTED_LANGUAGES
