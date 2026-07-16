"""
tests/preprocessing/test_language_detection.py

Tests unitaires pour src/preprocessing/language_detection.py — chaque
test correspond à un exemple vérifié manuellement pendant l'apprentissage.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from preprocessing.language_detection import (
    detect_language,
    detect_language_probabilities,
    is_supported_language,
)


def test_detects_english():
    assert (
        detect_language(
            "The delivery was very slow but the product quality is excellent"
        )
        == "en"
    )


def test_detects_spanish():
    assert (
        detect_language(
            "La entrega fue muy lenta pero la calidad del producto es excelente"
        )
        == "es"
    )


def test_detects_german():
    assert (
        detect_language(
            "Die Lieferung war sehr langsam, aber die Produktqualität ist ausgezeichnet"
        )
        == "de"
    )


def test_detects_french():
    assert (
        detect_language(
            "La livraison était très lente mais la qualité du produit est excellente"
        )
        == "fr"
    )


def test_detects_hindi():
    assert detect_language("डिलीवरी बहुत धीमी थी लेकिन उत्पाद की गुणवत्ता उत्कृष्ट है") == "hi"


def test_empty_text_returns_unknown_not_a_crash():
    # démontre le garde-fou : detect_langs() plante nativement sur texte
    # vide, notre wrapper ne doit pas laisser cette exception se propager
    assert detect_language("") == "unknown"


def test_probabilities_returns_tuples_not_language_objects():
    result = detect_language_probabilities("The delivery was great")
    assert isinstance(result, list)
    assert isinstance(result[0], tuple)
    lang, prob = result[0]
    assert isinstance(lang, str)
    assert isinstance(prob, float)


def test_is_supported_language():
    assert is_supported_language("The delivery was great") is True
