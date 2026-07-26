"""
tests/embeddings/test_glove_model.py

Tests unitaires pour src/embeddings/glove_model.py — utilise le modèle
pré-entraîné le plus petit (50 dimensions, ~66 Mo) pour limiter le temps
de téléchargement. Le cache module-level de glove_model.py garantit
qu'il n'est téléchargé/chargé qu'une seule fois pour tous ces tests.

Note : ces tests nécessitent une connexion internet (téléchargement du
modèle au premier appel) -- contrairement aux autres tests de ce projet
qui tournent entièrement hors-ligne une fois les dépendances installées.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from embeddings.glove import (
    get_word_vector,
    is_in_vocabulary,
    load_pretrained_glove,
    most_similar_words,
)


def _get_test_model():
    return load_pretrained_glove("glove-50")


def test_loads_large_real_vocabulary():
    model = _get_test_model()
    assert len(model.key_to_index) > 100_000  # vrai corpus, pas un jouet


def test_word_vector_has_correct_dimension():
    model = _get_test_model()
    vector = get_word_vector(model, "delivery")
    assert vector.shape == (50,)


def test_similar_words_are_semantically_related():
    model = _get_test_model()
    neighbors = most_similar_words(model, "delivery", topn=5)
    neighbor_words = [word for word, score in neighbors]
    # sur un vrai corpus, on s'attend a des mots du meme champ semantique
    assert any(
        w in neighbor_words for w in ["deliveries", "fast", "shipping", "delivered"]
    )


def test_unknown_word_raises_key_error():
    # contrairement a FastText, GloVe n'a pas de secours par n-grams
    model = _get_test_model()
    assert is_in_vocabulary(model, "asdkjfhaslkdjfh12345") is False
    try:
        get_word_vector(model, "asdkjfhaslkdjfh12345")
        assert False, "aurait du lever une KeyError"
    except KeyError:
        pass


def test_load_glove_from_local_file():
    # simule un extrait du format brut telecharge depuis nlp.stanford.edu
    # (mot + valeurs, SANS ligne d'en-tete -- contrairement au format Word2Vec)
    import tempfile

    import numpy as np

    from embeddings.glove import load_glove_from_file

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(
            "delivery " + " ".join(str(round(x, 4)) for x in np.random.randn(50)) + "\n"
        )
        f.write(
            "shipping " + " ".join(str(round(x, 4)) for x in np.random.randn(50)) + "\n"
        )
        temp_path = f.name

    model = load_glove_from_file(temp_path)
    assert len(model.key_to_index) == 2
    assert model["delivery"].shape == (50,)
