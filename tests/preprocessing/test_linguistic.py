"""
tests/preprocessing/test_linguistic.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from nltk import word_tokenize

from preprocessing.linguistic import (
    extract_noun_phrases,
    named_entities_nltk,
    pos_tag_tokens,
)


def test_pos_tag_identifies_nouns_and_adjectives():
    tokens = word_tokenize("The delivery was slow")
    tagged = pos_tag_tokens(tokens)
    tagged_dict = dict(tagged)
    assert tagged_dict["delivery"] == "NN"
    assert tagged_dict["slow"] == "JJ"


def test_extract_noun_phrases_merges_consecutive_nouns():
    tokens = word_tokenize("the product quality is excellent")
    phrases = extract_noun_phrases(tokens)
    assert "product quality" in phrases


def test_extract_noun_phrases_single_noun():
    tokens = word_tokenize("the delivery was slow")
    phrases = extract_noun_phrases(tokens)
    assert "delivery" in phrases


def test_named_entities_nltk_detects_person():
    tokens = word_tokenize("John from customer service was rude")
    entities = named_entities_nltk(tokens)
    entity_texts = [text for text, label in entities]
    assert "John" in entity_texts
