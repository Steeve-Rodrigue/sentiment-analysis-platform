"""
src/embeddings/contextual_embeddings.py

Phase 2 — Représentation du texte — bloc "embeddings de phrase et contextuels".

Théorie résumée (voir la conversation associée pour le détail) :

- Sentence embeddings (SBERT) : un modèle entraîné spécifiquement pour
  produire UN vecteur par phrase entière, tel que des phrases
  sémantiquement proches aient des vecteurs proches -- plus riche qu'une
  simple moyenne de vecteurs de mots, qui perd l'ordre des mots et
  traite chaque mot comme également important.

- Contextual embeddings (BERT et famille) : contrairement à Word2Vec/
  FastText/GloVe (vecteurs STATIQUES -- un mot = toujours le même
  vecteur), un modèle contextuel recalcule le vecteur d'un mot à chaque
  phrase, en fonction de son contexte précis. "bank" dans "sat by the
  bank" (rivière) et "deposited at the bank" (banque) obtient deux
  vecteurs DIFFÉRENTS.

Ce module NÉCESSITE un accès réseau à huggingface.co pour télécharger
les modèles -- non disponible dans l'environnement sandbox utilisé pour
écrire ce code (voir scripts/verify_contextual_embeddings.py pour le
vérifier vous-même en local).
"""

from __future__ import annotations

import numpy as np
import torch

from transformers import AutoModel, AutoTokenizer


def get_sentence_embeddings(
    sentences: list[str], model_name: str = "all-MiniLM-L6-v2"
) -> np.ndarray:
    """Encode une liste de phrases en un vecteur dense par phrase, via
    Sentence-BERT. Retourne un tableau (nb_phrases, dimension)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return model.encode(sentences)


def sentence_similarity(
    sentence_a: str, sentence_b: str, model_name: str = "all-MiniLM-L6-v2"
) -> float:
    """Similarité cosinus entre deux phrases, via leurs sentence
    embeddings -- répond directement au besoin "text similarity" du
    Part 1 (comparer deux avis dans leur ensemble, pas juste des mots)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embeddings = model.encode([sentence_a, sentence_b])
    a, b = embeddings[0], embeddings[1]
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def get_contextual_word_vectors(
    sentence: str, model_name: str = "bert-base-uncased"
) -> dict:
    """Retourne le vecteur contextuel de chaque token d'une phrase, via
    BERT. Contrairement à get_word_vector() (Word2Vec/FastText/GloVe),
    le même mot dans deux phrases différentes donnera deux vecteurs
    différents ici."""

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    tokens = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        output = model(**tokens)

    token_strings = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])
    vectors = output.last_hidden_state[0]  # (nb_tokens, dimension)

    return {tok: vec.numpy() for tok, vec in zip(token_strings, vectors)}


# Modèles comparés par défaut : un anglais-seul rapide, un anglais-seul plus
# gros/précis, et un multilingue (le plus pertinent pour ce projet EN/ES/DE/HI/FR).
DEFAULT_MODELS_TO_COMPARE = [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "paraphrase-multilingual-MiniLM-L12-v2",
]


def compare_sentence_models(
    sentences: list[str], model_names: list[str] | None = None
) -> dict:
    """Compare plusieurs modèles sentence-transformers sur les mêmes
    phrases : dimension du vecteur produit et temps d'encodage. Sert à
    choisir objectivement un modèle plutôt que d'en prendre un au
    hasard -- notamment pour la Phase 9, où le modèle doit fonctionner
    sur les 5 langues du projet, pas seulement l'anglais."""
    import time

    from sentence_transformers import SentenceTransformer

    model_names = model_names or DEFAULT_MODELS_TO_COMPARE
    results = {}
    for name in model_names:
        model = SentenceTransformer(name)
        start = time.perf_counter()
        embeddings = model.encode(sentences)
        elapsed = time.perf_counter() - start
        results[name] = {
            "dimension": embeddings.shape[1],
            "encode_time_seconds": round(elapsed, 4),
        }
    return results


def cross_lingual_similarity_test(
    model_name: str, sentence_en: str, sentence_other_lang: str
) -> float:
    """Teste la capacité MULTILINGUE d'un modèle : encode une phrase
    anglaise et sa traduction dans une autre langue, et retourne leur
    similarité. Un modèle anglais-seul (ex. all-MiniLM-L6-v2) donnera un
    score bas même si les phrases veulent dire la même chose -- un vrai
    modèle multilingue (ex. paraphrase-multilingual-MiniLM-L12-v2)
    donnera un score élevé, preuve qu'il comprend que les deux phrases
    sont sémantiquement équivalentes malgré la langue différente."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embeddings = model.encode([sentence_en, sentence_other_lang])
    a, b = embeddings[0], embeddings[1]
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
