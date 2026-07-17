"""
tests/embeddings/test_vectorizers.py

Tests unitaires pour src/embeddings/vectorizers.py — chaque test
correspond à un exemple vérifié manuellement pendant l'apprentissage.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from embeddings.vectorizers import build_bow, build_ngram_vectorizer, build_tfidf


def test_bow_counts_repeated_words():
    documents = ["delivery delivery delivery was fast"]
    vectors, vectorizer = build_bow(documents)
    vocab = list(vectorizer.get_feature_names_out())
    delivery_index = vocab.index("delivery")
    assert vectors.toarray()[0][delivery_index] == 3


def test_bow_ignores_word_order():
    # démontre la limite connue de BoW : ordre différent, même vecteur
    doc_a = ["the delivery was slow"]
    doc_b = ["slow was the delivery"]
    vectors_a, vec_a = build_bow(doc_a)
    vectors_b, vec_b = build_bow(doc_b)
    assert sorted(vec_a.get_feature_names_out()) == sorted(
        vec_b.get_feature_names_out()
    )
    assert (vectors_a.toarray() == vectors_b.toarray()).all()


def test_tfidf_downweights_common_words_vs_rare_words():
    documents = [
        "the delivery was slow",
        "the product quality is great",
        "the delivery delivery delivery was fast",
    ]
    vectors, vectorizer = build_tfidf(documents)
    vocab = list(vectorizer.get_feature_names_out())
    arr = vectors.toarray()

    the_index = vocab.index("the")  # apparaît dans les 3 documents
    delivery_index = vocab.index(
        "delivery"
    )  # apparaît dans 2 documents, 3 fois dans le 3e

    # "delivery" doit avoir un score TF-IDF plus élevé que "the" dans le 3e document
    assert arr[2][delivery_index] > arr[2][the_index]


def test_ngram_vectorizer_captures_negation_context():
    documents = [
        "the delivery was not good",
        "the delivery was very good",
    ]
    vectors, vectorizer = build_ngram_vectorizer(
        documents, ngram_range=(1, 2), use_tfidf=False
    )
    vocab = list(vectorizer.get_feature_names_out())

    assert "not good" in vocab
    assert "very good" in vocab

    not_good_index = vocab.index("not good")
    very_good_index = vocab.index("very good")
    arr = vectors.toarray()

    # le document 1 doit avoir "not good" mais pas "very good", et inversement
    assert arr[0][not_good_index] == 1
    assert arr[0][very_good_index] == 0
    assert arr[1][not_good_index] == 0
    assert arr[1][very_good_index] == 1


def test_ngram_vectorizer_default_uses_tfidf():
    documents = ["the delivery was slow", "the product is great"]
    vectors, vectorizer = build_ngram_vectorizer(documents)
    from sklearn.feature_extraction.text import TfidfVectorizer

    assert isinstance(vectorizer, TfidfVectorizer)
