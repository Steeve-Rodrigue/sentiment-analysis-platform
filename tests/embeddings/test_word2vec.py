"""
tests/embeddings/test_word2vec_model.py

Tests unitaires pour src/embeddings/word2vec_model.py — le corpus est
volontairement répété (x50) car Word2Vec a besoin d'une exposition
répétée aux mêmes patterns de co-occurrence pour produire des vecteurs
fiables (voir la conversation associée : sur un corpus non-répété, les
résultats sont proches du bruit).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from embeddings.word2vec import get_word_vector, most_similar_words, train_word2vec

_PHRASES_BASE = [
    "the delivery was fast".split(),
    "the delivery was slow".split(),
    "the shipping was fast".split(),
    "the shipping was slow".split(),
    "the delivery arrived quickly".split(),
    "the shipping arrived quickly".split(),
    "the product quality was great".split(),
    "the product quality was poor".split(),
]
_PHRASES = _PHRASES_BASE * 50


def _train_test_model():
    return train_word2vec(_PHRASES, vector_size=50, window=3, epochs=100)


def test_word_vector_has_correct_dimension():
    model = _train_test_model()
    vector = get_word_vector(model, "delivery")
    assert vector.shape == (50,)


def test_similar_context_words_end_up_close_in_vector_space():
    # "delivery" et "shipping" apparaissent dans des contextes identiques
    # dans le corpus jouet -> doivent être les voisins les plus proches
    model = _train_test_model()
    neighbors = most_similar_words(model, "delivery", topn=3)
    neighbor_words = [word for word, score in neighbors]
    assert "shipping" in neighbor_words


def test_most_similar_returns_scores_between_minus_one_and_one():
    # la similarité cosinus est toujours dans [-1, 1]
    model = _train_test_model()
    neighbors = most_similar_words(model, "delivery", topn=3)
    for word, score in neighbors:
        assert -1.0 <= score <= 1.0
