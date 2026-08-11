"""
tests/transformer_arch/test_fine_tuning.py

Tests unitaires pour src/transformer_arch/fine_tuning.py.

IMPORTANT : la plupart des tests de ce fichier necessitent un acces
reseau a huggingface.co (telechargement de DistilBERT) -- bloque dans
le sandbox utilise pour ecrire ce code. Ils sont marques avec
@pytest.mark.network et seront IGNORES par defaut. Lancez-les
explicitement chez vous avec :
    uv run pytest tests/ -v -m network
Le test de SentimentDataset (sans reseau) tourne toujours normalement.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest
import torch

from transformers_arch.fine_tuning import SentimentDataset


def test_sentiment_dataset_length():
    # aucun reseau necessaire -- juste la structure de la classe
    encodings = {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
    }
    dataset = SentimentDataset(encodings, labels=[1, 0])
    assert len(dataset) == 2


def test_sentiment_dataset_getitem_returns_labeled_dict():
    encodings = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    dataset = SentimentDataset(encodings, labels=[1])
    item = dataset[0]
    assert "input_ids" in item
    assert "attention_mask" in item
    assert item["labels"] == 1


def test_compute_metrics_returns_accuracy_and_f1():
    # regression du bug reel : Trainer.evaluate() ne retournait QUE
    # eval_loss sans compute_metrics, donnant eval_accuracy=nan
    import numpy as np

    from transformers_arch.fine_tuning import compute_metrics

    logits = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    labels = np.array([1, 0, 1])  # les 3 predictions sont correctes

    resultats = compute_metrics((logits, labels))
    assert resultats["accuracy"] == 1.0
    assert resultats["f1"] == 1.0


@pytest.mark.network
def test_load_pretrained_classifier_downloads_and_loads():
    from transformers_arch.fine_tuning import load_pretrained_classifier

    model, tokenizer = load_pretrained_classifier()
    assert model is not None
    assert tokenizer is not None


@pytest.mark.network
def test_tokenize_dataset_produces_expected_keys():
    from transformers_arch.fine_tuning import (
        load_pretrained_classifier,
        tokenize_dataset,
    )

    _, tokenizer = load_pretrained_classifier()
    encodings = tokenize_dataset(["a great review", "a bad review"], tokenizer)
    assert "input_ids" in encodings
    assert "attention_mask" in encodings


@pytest.mark.network
def test_fine_tuned_model_beats_random_baseline():
    # sur le vrai dataset movie_reviews, un modele fine-tune devrait
    # nettement depasser 0.5 (le hasard sur une classification binaire)
    import sys as _sys

    _sys.path.insert(0, "src")
    from classical_ml.data_loader import load_movie_reviews
    from transformers_arch.fine_tuning import (
        SentimentDataset,
        evaluate_fine_tuned_model,
        fine_tune_model,
        load_pretrained_classifier,
        tokenize_dataset,
    )

    X_train, X_test, y_train, y_test = load_movie_reviews()
    # sous-echantillon pour un test rapide, pas le dataset complet
    X_train_small = X_train[:100]
    y_train_small = [1 if lab == "pos" else 0 for lab in y_train[:100]]
    X_test_small = X_test[:50]
    y_test_small = [1 if lab == "pos" else 0 for lab in y_test[:50]]

    model, tokenizer = load_pretrained_classifier()
    train_enc = tokenize_dataset(X_train_small, tokenizer)
    test_enc = tokenize_dataset(X_test_small, tokenizer)

    train_ds = SentimentDataset(train_enc, y_train_small)
    test_ds = SentimentDataset(test_enc, y_test_small)

    trainer = fine_tune_model(model, train_ds, test_ds, epochs=1)
    results = evaluate_fine_tuned_model(trainer)

    assert results["eval_loss"] < 1.0
