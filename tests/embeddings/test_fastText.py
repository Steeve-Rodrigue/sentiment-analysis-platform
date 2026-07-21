"""
tests/embeddings/test_fasttext_model.py

Tests unitaires pour src/embeddings/fasttext_model.py — le test le plus
important démontre la différence fondamentale avec Word2Vec : produire
un vecteur pour un mot jamais vu à l'entraînement.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from embeddings.fastText import (
    get_word_vector,
    is_out_of_vocabulary,
    most_similar_words,
    train_fasttext,
)

_PHRASES_BASE = [
    "the delivery was fast".split(),
    "the delivery was slow".split(),
    "the shipping was fast".split(),
    "the shipping was slow".split(),
    "the delivery arrived quickly".split(),
    "the shipping arrived quickly".split(),
] * 50


def _train_test_model():
    return train_fasttext(_PHRASES_BASE, vector_size=50, window=3, epochs=100)


def test_word_vector_has_correct_dimension():
    model = _train_test_model()
    vector = get_word_vector(model, "delivery")
    assert vector.shape == (50,)


def test_similar_context_words_end_up_close():
    model = _train_test_model()
    neighbors = most_similar_words(model, "delivery", topn=3)
    neighbor_words = [word for word, score in neighbors]
    assert "shipping" in neighbor_words


def test_produces_vector_for_word_never_seen_in_training():
    # démontre la différence fondamentale avec Word2Vec : ce mot précis
    # ("deliveryyy") n'existe dans AUCUNE phrase du corpus d'entraînement
    model = _train_test_model()
    assert is_out_of_vocabulary(model, "deliveryyy")
    vector = get_word_vector(model, "deliveryyy")  # ne doit PAS lever de KeyError
    assert vector.shape == (50,)


def test_oov_word_is_closest_to_its_root_word():
    # "deliveryyy" partage la plupart de ses n-grams de caracteres avec
    # "delivery" -> doit ressortir comme voisin le plus proche
    model = _train_test_model()
    neighbors = most_similar_words(model, "deliveryyy", topn=1)
    assert neighbors[0][0] == "delivery"
    assert neighbors[0][1] > 0.9  # tres forte similarite attendue


def test_known_word_is_not_out_of_vocabulary():
    model = _train_test_model()
    assert is_out_of_vocabulary(model, "delivery") is False
