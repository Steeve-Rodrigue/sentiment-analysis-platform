"""
src/llms/rag.py

Phase 7 -- LLMs -- bloc "RAG (Retrieval-Augmented Generation)".

Theorie resumee (voir la conversation associee pour le detail) :

Un LLM a une connaissance FIGEE au moment de son entrainement -- il ne
connait rien des avis clients specifiques du projet. RAG combine deux
etapes deja construites separement dans ce projet :

1. RETRIEVAL : utilise les embeddings de phrase (Phase 2,
   sentence-transformers) pour trouver, parmi une base de documents,
   ceux SEMANTIQUEMENT proches de la question posee -- similarite
   cosinus entre le vecteur de la question et ceux des documents.
2. GENERATION : donne ces documents retrouves EN CONTEXTE au LLM, qui
   repond en s'appuyant dessus plutot que sur sa memoire figee.

Verifie empiriquement (vecteurs synthetiques) : sur une question sur
la "livraison", les documents parlant de livraison/service (0.998,
0.997 de similarite) sont bien retrouves en priorite, loin devant un
document sans rapport comme l'emballage (0.071).

La partie qu'on peut tester sans reseau : la recherche par
similarite (pure algebre) ET l'ingestion de PDF (extraction + chunking,
pure manipulation de texte, aucun modele necessaire).

--- INGESTION DE PDF ---

Un vrai document source (rapport, export d'avis clients) ne peut pas
etre traite comme UN SEUL document pour le retrieval -- trop long,
trop de sujets differents melanges. On le decoupe en PASSAGES
("chunks") plus petits et coherents, chacun devenant une entree
independante de la base a interroger.

Le CHEVAUCHEMENT (overlap) entre chunks consecutifs evite qu'une
information a cheval sur la coupure soit perdue -- verifie
empiriquement sur un PDF de test : le texte "next day. Customer
Review 2:" apparait a la fois en fin du chunk 0 et en debut du
chunk 1, garantissant qu'aucune requete portant sur cette frontiere
ne rate l'information.
"""

from __future__ import annotations

import numpy as np


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extrait tout le texte d'un PDF, page par page. Utilise pypdf
    (extraction basique, texte pur) -- suffisant pour des documents
    textuels (rapports, avis clients exportes). Pour des PDF avec
    tableaux/mise en page complexe, voir pdfplumber (skill pdf-reading)."""
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    """Decoupe un texte en chunks de chunk_size MOTS, avec overlap
    mots de chevauchement entre chunks consecutifs -- evite qu'une
    information a cheval sur une coupure soit perdue."""
    words = text.split()
    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += step
    return chunks


def load_pdf_as_documents(
    pdf_path: str, chunk_size: int = 200, overlap: int = 50
) -> list[str]:
    """Pipeline complet d'ingestion : extrait le texte d'un PDF puis
    le decoupe en chunks, prets a etre encodes par
    build_document_index() comme n'importe quelle liste de documents."""
    text = extract_text_from_pdf(pdf_path)
    return chunk_text(text, chunk_size=chunk_size, overlap=overlap)


def find_most_similar(
    query_embedding: np.ndarray,
    document_embeddings: np.ndarray,
    top_k: int = 3,
) -> list[tuple[int, float]]:
    """Retourne les indices des top_k documents les plus proches de la
    question (par similarite cosinus), avec leur score. Pure algebre,
    testable sans aucun modele."""
    similarities = (
        document_embeddings
        @ query_embedding
        / (
            np.linalg.norm(document_embeddings, axis=1)
            * np.linalg.norm(query_embedding)
        )
    )
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [(int(i), float(similarities[i])) for i in top_indices]


def build_document_index(documents: list[str], model_name: str = None):
    """Encode une liste de documents en embeddings de phrase, via
    sentence-transformers (Phase 2). Necessite un acces reseau pour
    telecharger le modele -- a lancer chez vous."""
    from sentence_transformers import SentenceTransformer

    model_name = model_name or "all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    embeddings = model.encode(documents)
    return embeddings, model


def retrieve_relevant_documents(
    query: str,
    documents: list[str],
    document_embeddings: np.ndarray,
    embedding_model,
    top_k: int = 3,
) -> list[tuple[str, float]]:
    """Pipeline complet de retrieval : encode la question avec le meme
    modele que les documents, puis retourne les documents les plus
    proches avec leur score."""
    query_embedding = embedding_model.encode(query)
    top_matches = find_most_similar(query_embedding, document_embeddings, top_k)
    return [(documents[i], score) for i, score in top_matches]


def build_rag_prompt(query: str, retrieved_documents: list[str]) -> str:
    """Construit le prompt final combinant la question et les
    documents retrouves -- le LLM doit repondre EN S'APPUYANT sur ce
    contexte, pas sur sa memoire figee."""
    context = "\n".join(f"- {doc}" for doc in retrieved_documents)
    return (
        f"Context (customer reviews):\n{context}\n\n"
        f"Question: {query}\n"
        f"Answer based only on the context above:"
    )


def answer_with_rag(
    query: str,
    documents: list[str],
    document_embeddings: np.ndarray,
    embedding_model,
    generate_fn,
    top_k: int = 3,
) -> dict:
    """Pipeline RAG complet : retrieval puis generation. `generate_fn`
    est une fonction prompt -> str (ex. classify_with_llm de
    prompting.py, ou toute autre fonction d'appel a un LLM)."""
    retrieved = retrieve_relevant_documents(
        query, documents, document_embeddings, embedding_model, top_k
    )
    retrieved_texts = [doc for doc, _ in retrieved]
    prompt = build_rag_prompt(query, retrieved_texts)
    answer = generate_fn(prompt)
    return {
        "answer": answer,
        "retrieved_documents": retrieved,
        "prompt": prompt,
    }
