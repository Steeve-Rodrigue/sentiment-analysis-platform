"""
tests/llms/test_prompting.py

Tests unitaires pour src/llms/prompting.py.

La plupart des tests ne necessitent AUCUN reseau (les prompts sont du
texte pur). Seul classify_with_llm est marque @pytest.mark.network,
car il telecharge un vrai modele generatif.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

from llms.prompting import (
    build_chain_of_thought_prompt,
    build_few_shot_prompt,
    build_zero_shot_prompt,
)


def test_zero_shot_prompt_contains_review_text():
    prompt = build_zero_shot_prompt("great delivery")
    assert "great delivery" in prompt
    assert "Sentiment:" in prompt


def test_zero_shot_prompt_uses_default_labels():
    prompt = build_zero_shot_prompt("great delivery")
    assert "positive" in prompt
    assert "negative" in prompt
    assert "mixed" in prompt


def test_zero_shot_prompt_accepts_custom_labels():
    prompt = build_zero_shot_prompt("great delivery", labels=("good", "bad"))
    assert "good" in prompt
    assert "bad" in prompt
    assert "mixed" not in prompt


def test_few_shot_prompt_includes_all_examples():
    exemples = [
        ("Amazing quality!", "positive"),
        ("Arrived broken", "negative"),
    ]
    prompt = build_few_shot_prompt("slow delivery", exemples)
    assert "Amazing quality!" in prompt
    assert "Arrived broken" in prompt
    assert "slow delivery" in prompt


def test_few_shot_prompt_puts_query_last():
    exemples = [("Amazing quality!", "positive")]
    prompt = build_few_shot_prompt("slow delivery", exemples)
    lignes = prompt.split("\n")
    assert "slow delivery" in lignes[-1]
    assert lignes[-1].endswith("Sentiment:")


def test_few_shot_prompt_with_no_examples_still_works():
    prompt = build_few_shot_prompt("slow delivery", [])
    assert "slow delivery" in prompt


def test_chain_of_thought_prompt_asks_for_reasoning():
    prompt = build_chain_of_thought_prompt("mixed review here")
    assert "step by step" in prompt
    assert "aspects" in prompt.lower()
    assert "mixed review here" in prompt


@pytest.mark.network
def test_classify_with_llm_returns_text():
    from llms.prompting import build_zero_shot_prompt, classify_with_llm

    prompt = build_zero_shot_prompt(
        "The delivery was slow but the product quality is excellent"
    )
    result = classify_with_llm(prompt)
    assert isinstance(result, str)
    assert len(result) > 0
