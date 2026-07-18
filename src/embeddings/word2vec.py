"""
src/embeddings/word2vec_model.py

Phase 2 — Représentation du texte — bloc "Word2Vec".

Théorie résumée (voir la conversation associée pour le détail) :
Contrairement à BoW/TF-IDF (comptage, vecteurs creux), Word2Vec APPREND
des vecteurs denses en entraînant un petit réseau à prédire un mot à
partir de son contexte (CBOW) ou l'inverse (Skip-gram). Le sous-produit
de cet entraînement -- les poids internes appris -- devient le vecteur
du mot. Des mots utilisés dans des contextes similaires ("delivery",
"shipping") finissent géométriquement proches dans l'espace vectoriel.

LIMITE IMPORTANTE, vérifiée empiriquement : Word2Vec est très gourmand en
données. Sur un corpus jouet de 10 phrases, les vecteurs obtenus sont
quasiment du bruit (similarités < 0.2, mauvais voisins). Il faut des
millions de mots pour des vecteurs fiables -- en pratique, on utilise
presque toujours des vecteurs PRÉ-ENTRAÎNÉS plutôt que d'entraîner
Word2Vec sur son propre petit dataset (voir Phase 2, comparaison avec
les modèles Hugging Face à venir).

sg=1 -> Skip-gram (meilleur sur mots rares, en théorie -- effet visible
        surtout sur de vrais corpus larges et déséquilibrés, pas sur de
        petits exemples jouets)
sg=0 -> CBOW (entraînement plus rapide)
"""

from __future__ import annotations

from gensim.models import Word2Vec


def train_word2vec(
    tokenized_documents: list[list[str]],
    vector_size: int = 100,
    window: int = 5,
    sg: int = 1,
    epochs: int = 20,
    min_count: int = 1,
    seed: int = 42,
) -> Word2Vec:
    """Entraîne un modèle Word2Vec sur des documents déjà tokenisés
    (liste de listes de tokens -- utiliser tokenize_document() de
    src/preprocessing/tokenization.py pour préparer l'entrée)."""
    return Word2Vec(
        sentences=tokenized_documents,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        epochs=epochs,
        seed=seed,
    )


def most_similar_words(
    model: Word2Vec, word: str, topn: int = 5
) -> list[tuple[str, float]]:
    """Retourne les topn mots les plus proches d'un mot donné dans
    l'espace vectoriel appris, avec leur score de similarité cosinus."""
    return model.wv.most_similar(word, topn=topn)


def get_word_vector(model: Word2Vec, word: str):
    """Retourne le vecteur dense appris pour un mot donné."""
    return model.wv[word]
