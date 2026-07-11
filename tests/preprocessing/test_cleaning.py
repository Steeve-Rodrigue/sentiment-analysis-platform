"""
tests/preprocessing/test_cleaning.py

Tests unitaires pour src/preprocessing/cleaning.py — chaque test correspond
à un exemple qu'on a vérifié manuellement pendant l'apprentissage.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from preprocessing.cleaning import (
    normalize_unicode,
    remove_urls,
    process_hashtags,
    demojize,
    reduce_repeated_chars,
    normalize_repeated_punctuation,
    clean_text,
)


def test_normalize_unicode_makes_nfc_and_nfd_equal():
    nfc = "café"
    nfd = "cafe\u0301"  # "e" + accent combinant
    assert nfc != nfd  # avant normalisation : différents
    assert normalize_unicode(nfc) == normalize_unicode(nfd)  # après : égaux


def test_remove_urls_strips_both_url_forms():
    assert remove_urls("voir https://amazon.fr/produit/123 svp").strip() == "voir  svp".strip()
    assert "www.site.com" not in remove_urls("aller sur www.site.com maintenant")


def test_remove_urls_does_not_touch_text_without_url():
    text = "Livraison lente cette fois-ci"
    assert remove_urls(text) == text


def test_process_hashtags_split_camel():
    result = process_hashtags("Service #GreatService ici")
    assert result == "Service Great Service ici"


def test_demojize_converts_emoji_to_token():
    result = demojize("Livraison rapide 👍")
    assert "__EMOJI_thumbs_up__" in result.replace(" ", "")


def test_reduce_repeated_chars():
    assert reduce_repeated_chars("sooooo") == "soo"
    assert reduce_repeated_chars("nullll") == "null"


def test_normalize_repeated_punctuation():
    assert normalize_repeated_punctuation("incroyable!!!!") == "incroyable!"


def test_clean_text_end_to_end_on_noisy_review():
    raw = (
        "@GlobaTrend Livraison sooooo lente !!!! 😡 mais le produit "
        "est top #GreatService, voir https://amazon.fr/produit/123 pour + d'infos"
    )
    result = clean_text(raw)
    # ce qui doit avoir disparu
    assert "@globatrend" not in result
    assert "https://" not in result
    # ce qui doit avoir été transformé et préservé
    assert "great service" in result
    assert "soo" in result
    assert "__emoji_enraged_face__" in result