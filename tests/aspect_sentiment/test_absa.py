"""
tests/aspect_sentiment/test_absa.py

Tests unitaires pour src/aspect_sentiment/absa.py.

extract_aspect_candidates ne necessite AUCUN reseau (reutilise NLTK,
Phase 1). Le reste necessite huggingface.co (DistilBERT) -- marque
@pytest.mark.network.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

from aspect_sentiment.absa import (
    ASPECT_CATEGORIES,
    AspectPairDataset,
    extract_aspect_candidates,
)


def test_aspect_categories_match_project_scope():
    # rappel plan-projet-globatrend-insights.md : 9 aspects definis
    assert len(ASPECT_CATEGORIES) == 9
    assert "delivery" in ASPECT_CATEGORIES
    assert "product_quality" in ASPECT_CATEGORIES


def test_extract_aspect_candidates_finds_real_aspects():
    avis = "The delivery was slow but the product quality is excellent"
    aspects = extract_aspect_candidates(avis)
    assert "delivery" in aspects
    assert "product quality" in aspects


def test_extract_aspect_candidates_on_multi_aspect_review():
    avis = (
        "The delivery was slow but the product quality is excellent. "
        "Customer service was also very helpful."
    )
    aspects = extract_aspect_candidates(avis)
    assert len(aspects) >= 3


def test_aspect_pair_dataset_length():
    import torch

    encodings = {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
        "token_type_ids": torch.tensor([[0, 0, 1], [0, 0, 1]]),
    }
    dataset = AspectPairDataset(encodings, labels=[2, 0])
    assert len(dataset) == 2


def test_aspect_pair_dataset_getitem_returns_labeled_dict():
    import torch

    encodings = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "token_type_ids": torch.tensor([[0, 0, 1]]),
    }
    dataset = AspectPairDataset(encodings, labels=[2])
    item = dataset[0]
    assert item["labels"] == 2
    assert "token_type_ids" in item


@pytest.mark.network
def test_tokenize_aspect_pairs_produces_token_type_ids():
    from aspect_sentiment.absa import (
        load_absa_classifier,
        tokenize_aspect_pairs,
    )

    _, tokenizer = load_absa_classifier()
    encodings = tokenize_aspect_pairs(
        ["The delivery was slow"], ["delivery"], tokenizer
    )
    assert "token_type_ids" in encodings
    # au moins deux segments distincts (0 pour le texte, 1 pour l'aspect)
    assert encodings["token_type_ids"].max().item() >= 1


@pytest.mark.network
def test_predict_aspect_sentiment_returns_one_label_per_aspect():
    from aspect_sentiment.absa import (
        load_absa_classifier,
        predict_aspect_sentiment,
    )

    model, tokenizer = load_absa_classifier()
    avis = "The delivery was slow but the product quality is excellent"
    aspects = ["delivery", "product quality"]

    resultats = predict_aspect_sentiment(avis, aspects, model, tokenizer)
    assert set(resultats.keys()) == set(aspects)
    for label in resultats.values():
        assert label in ["negative", "neutral", "positive"]
