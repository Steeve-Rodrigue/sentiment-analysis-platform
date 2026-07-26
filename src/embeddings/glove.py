"""
src/embeddings/glove_model.py

Phase 2 — Représentation du texte — bloc "GloVe".

Théorie résumée (voir la conversation associée pour le détail) :
GloVe (Global Vectors) diffère de Word2Vec dans SA MÉTHODE
d'entraînement : au lieu d'apprendre depuis des fenêtres de contexte
locales glissantes, GloVe construit d'abord une matrice de
co-occurrence GLOBALE (comptage, sur tout le corpus, de combien de fois
chaque paire de mots apparaît ensemble), puis factorise cette matrice
en vecteurs denses tels que :
    w_i . w_j ~= log(X_ij)
où X_ij est le nombre de co-occurrences du mot i et du mot j.

Choix délibéré : ce module ne fournit PAS de fonction train_glove().
gensim n'offre pas d'entraîneur GloVe (construire la matrice de
co-occurrence globale soi-même est hors scope ici), et en pratique
personne n'entraîne GloVe from scratch sur son propre petit dataset --
on charge des vecteurs PRÉ-ENTRAÎNÉS (Stanford, sur Wikipedia +
Gigaword, ici via gensim-data). C'est fidèle à l'usage réel, pas une
simplification qui cache une lacune.
"""

from __future__ import annotations

import gensim.downloader as api
from gensim.models import KeyedVectors

_MODEL_CACHE: dict[str, KeyedVectors] = {}

AVAILABLE_MODELS = {
    "glove-50": "glove-wiki-gigaword-50",
    "glove-100": "glove-wiki-gigaword-100",
    "glove-200": "glove-wiki-gigaword-200",
    "glove-300": "glove-wiki-gigaword-300",
}


def load_pretrained_glove(model_name: str = "glove-50") -> KeyedVectors:
    """Charge (et met en cache) des vecteurs GloVe pré-entraînés via
    gensim.downloader -- télécharge automatiquement au premier appel."""
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Modèle inconnu '{model_name}'. Choix : {list(AVAILABLE_MODELS)}"
        )
    gensim_name = AVAILABLE_MODELS[model_name]
    if gensim_name not in _MODEL_CACHE:
        _MODEL_CACHE[gensim_name] = api.load(gensim_name)
    return _MODEL_CACHE[gensim_name]


def load_glove_from_file(path: str) -> KeyedVectors:
    """Charge des vecteurs GloVe depuis un fichier .txt déjà téléchargé
    manuellement (ex. glove.6B.50d.txt depuis nlp.stanford.edu/projects/glove).

    IMPORTANT : ces fichiers n'ont PAS de ligne d'en-tête (contrairement
    au format Word2Vec, qui commence par "nombre_de_mots dimension") --
    no_header=True est nécessaire, sinon gensim interprète à tort la
    première ligne de vecteurs comme des métadonnées et casse le
    chargement."""
    if path not in _MODEL_CACHE:
        _MODEL_CACHE[path] = KeyedVectors.load_word2vec_format(
            path, binary=False, no_header=True
        )
    return _MODEL_CACHE[path]


def most_similar_words(
    model: KeyedVectors, word: str, topn: int = 5
) -> list[tuple[str, float]]:
    """Retourne les topn mots les plus proches. Contrairement à FastText,
    échoue (KeyError) sur un mot hors-vocabulaire -- GloVe n'a pas de
    mécanisme de n-grams de caractères pour en reconstruire un."""
    return model.most_similar(word, topn=topn)


def get_word_vector(model: KeyedVectors, word: str):
    """Retourne le vecteur pré-entraîné d'un mot."""
    return model[word]


def is_in_vocabulary(model: KeyedVectors, word: str) -> bool:
    return word in model.key_to_index
