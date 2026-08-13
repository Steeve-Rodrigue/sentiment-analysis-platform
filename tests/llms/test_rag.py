"""
tests/llms/test_rag.py

Tests unitaires pour src/llms/rag.py.

find_most_similar et build_rag_prompt sont testables sans reseau (pure
algebre / texte). Les autres fonctions necessitent
sentence-transformers et sont marquees @pytest.mark.network.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
import pytest

from llms.rag import answer_with_rag, build_rag_prompt, find_most_similar


def test_find_most_similar_ranks_by_cosine_similarity():
    query = np.array([1.0, 0.0, 0.0])
    documents = np.array(
        [
            [1.0, 0.0, 0.0],  # identique -> similarite max
            [0.0, 1.0, 0.0],  # orthogonal -> similarite nulle
            [0.9, 0.1, 0.0],  # tres proche
        ]
    )
    resultats = find_most_similar(query, documents, top_k=3)
    indices = [i for i, _ in resultats]
    assert indices[0] == 0  # le document identique doit etre premier
    assert indices[1] == 2  # puis le plus proche


def test_find_most_similar_respects_top_k():
    query = np.array([1.0, 0.0])
    documents = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    resultats = find_most_similar(query, documents, top_k=2)
    assert len(resultats) == 2


def test_find_most_similar_scores_are_bounded():
    query = np.array([1.0, 2.0, 3.0])
    documents = np.random.randn(10, 3)
    resultats = find_most_similar(query, documents, top_k=10)
    for _, score in resultats:
        assert -1.0 <= score <= 1.0


def test_build_rag_prompt_includes_context_and_query():
    prompt = build_rag_prompt(
        "How is the delivery?",
        ["Delivery was fast", "Product quality is great"],
    )
    assert "Delivery was fast" in prompt
    assert "Product quality is great" in prompt
    assert "How is the delivery?" in prompt


def test_build_rag_prompt_with_no_documents_still_works():
    prompt = build_rag_prompt("Any question?", [])
    assert "Any question?" in prompt


def test_answer_with_rag_uses_injected_generate_function():
    # verifie que le pipeline complet appelle bien generate_fn avec le
    # prompt construit, sans avoir besoin d'un vrai LLM (generate_fn
    # injectee ici est une fonction jouet)
    documents = ["Delivery was fast", "Product was excellent"]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])

    class FakeEmbedder:
        def encode(self, text):
            return np.array([1.0, 0.0])  # toujours proche du doc 0

    def fake_generate(prompt):
        return f"FAKE ANSWER based on: {prompt[:20]}"

    resultat = answer_with_rag(
        "How is delivery?",
        documents,
        embeddings,
        FakeEmbedder(),
        fake_generate,
        top_k=1,
    )
    assert "FAKE ANSWER" in resultat["answer"]
    assert resultat["retrieved_documents"][0][0] == "Delivery was fast"


@pytest.mark.network
def test_build_document_index_downloads_and_encodes():
    from llms.rag import build_document_index

    embeddings, model = build_document_index(["Delivery was fast", "Product was great"])
    assert embeddings.shape[0] == 2
