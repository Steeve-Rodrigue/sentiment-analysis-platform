"""
src/embeddings/fasttext_model.py

Phase 2 — Représentation du texte — bloc "FastText".

Théorie résumée (voir la conversation associée pour le détail) :
FastText étend Word2Vec en représentant chaque mot comme un sac de
n-grams de CARACTÈRES (ex. "delivery" -> "<de", "del", "eli", "liv"...)
plutôt qu'un seul vecteur par mot entier. Le vecteur final d'un mot est
la somme des vecteurs de ses n-grams.

Bénéfice majeur, vérifié empiriquement : FastText peut produire un
vecteur pour un mot JAMAIS vu à l'entraînement (ex. une faute de frappe
"deliveryyy"), tant que ce mot partage des fragments de caractères avec
des mots connus -- Word2Vec, lui, lève une KeyError sur un mot inconnu.
Compromis : entraînement plus lent, modèle plus volumineux (vecteurs
supplémentaires pour chaque n-gram de caractères, pas juste les mots).

min_n / max_n contrôlent la taille des n-grams de caractères utilisés
(par défaut 3 à 6 caractères).
"""

from __future__ import annotations

from gensim.models import FastText


def train_fasttext(
    tokenized_documents: list[list[str]],
    vector_size: int = 100,
    window: int = 5,
    sg: int = 1,
    epochs: int = 20,
    min_count: int = 1,
    min_n: int = 3,
    max_n: int = 6,
    seed: int = 42,
) -> FastText:
    """Entraîne un modèle FastText sur des documents déjà tokenisés
    (même format d'entrée que train_word2vec())."""
    return FastText(
        sentences=tokenized_documents,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        epochs=epochs,
        min_n=min_n,
        max_n=max_n,
        seed=seed,
    )


def most_similar_words(
    model: FastText, word: str, topn: int = 5
) -> list[tuple[str, float]]:
    """Retourne les topn mots les plus proches -- fonctionne même si
    `word` n'a jamais été vu à l'entraînement (contrairement à Word2Vec),
    tant qu'il partage des n-grams de caractères avec le vocabulaire connu."""
    return model.wv.most_similar(word, topn=topn)


def get_word_vector(model: FastText, word: str):
    """Retourne le vecteur d'un mot -- reconstruit à partir des n-grams
    de caractères si le mot est hors-vocabulaire (OOV)."""
    return model.wv[word]


def is_out_of_vocabulary(model: FastText, word: str) -> bool:
    """Indique si un mot n'a jamais été vu tel quel à l'entraînement
    (même si FastText peut quand même lui construire un vecteur via ses
    n-grams de caractères)."""
    return word not in model.wv.key_to_index
