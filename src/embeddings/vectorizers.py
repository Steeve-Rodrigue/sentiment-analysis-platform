"""
src/embeddings/vectorizers.py

Phase 2 — Représentation du texte — bloc "Bag of Words, TF-IDF, n-grams".

Théorie résumée (voir la conversation associée pour le détail) :
- Bag of Words (BoW) : compte les occurrences de chaque mot, ignore
  totalement l'ordre des mots. Simple mais deux phrases avec les mêmes
  mots dans un ordre différent produisent le même vecteur.
- TF-IDF : pondère chaque comptage par la rareté du mot dans l'ensemble
  des documents (TF * IDF), pour que des mots omniprésents ("the") ne
  dominent pas le vecteur face à des mots rares mais informatifs.
- N-grams : au lieu de mots isolés, utilise des séquences de N mots
  consécutifs comme unités de vocabulaire -- "not good" et "very good"
  deviennent des entrées de vocabulaire distinctes, capturant une partie
  de l'ordre des mots que les unigrammes seuls perdent complètement.

Ces trois techniques partagent la même interface scikit-learn
(fit_transform / get_feature_names_out), volontairement -- ça permet de
passer de l'une à l'autre en changeant un seul paramètre.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def build_bow(documents: list[str], ngram_range: tuple[int, int] = (1, 1)):
    """Bag of Words : comptages bruts d'occurrence par mot (ou n-gram)."""
    vectorizer = CountVectorizer(ngram_range=ngram_range)
    vectors = vectorizer.fit_transform(documents)
    return vectors, vectorizer


def build_tfidf(documents: list[str], ngram_range: tuple[int, int] = (1, 1)):
    """TF-IDF : comptages pondérés par la rareté du mot (ou n-gram) dans
    l'ensemble des documents -- réduit l'importance des mots omniprésents
    sans information discriminative (ex. "the")."""
    vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    vectors = vectorizer.fit_transform(documents)
    return vectors, vectorizer


def build_ngram_vectorizer(
    documents: list[str],
    ngram_range: tuple[int, int] = (1, 2),
    use_tfidf: bool = True,
):
    """Point d'entrée générique : choisit BoW ou TF-IDF, avec une plage
    de n-grams configurable. ngram_range=(1, 2) = unigrammes + bigrammes,
    un bon compromis par défaut (au-delà, le vocabulaire explose vite
    sans forcément apporter plus de signal utile)."""
    VectorizerClass = TfidfVectorizer if use_tfidf else CountVectorizer
    vectorizer = VectorizerClass(ngram_range=ngram_range)
    vectors = vectorizer.fit_transform(documents)
    return vectors, vectorizer
