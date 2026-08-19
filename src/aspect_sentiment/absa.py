"""
src/aspect_sentiment/absa.py

Phase 9 -- ABSA (produit central du projet).

Deux architectures possibles pour l'ABSA :
1. PIPELINE : extraction d'aspects (etape separee) PUIS classification
   du sentiment pour chaque aspect trouve.
2. JOINTE (retenue ici) : reformule le probleme comme une
   classification de PAIRE DE PHRASES -- (texte, aspect_candidat) ->
   sentiment. Meme mecanisme technique que DistilBERT (Phase 6), mais
   avec une entree a deux segments au lieu d'un seul.

L'EXTRACTION D'ASPECTS reutilise extract_noun_phrases() de la Phase 1
(preprocessing/linguistic.py) -- verifie empiriquement sur un avis
multi-aspects reel : les 3 aspects reels ("delivery", "product
quality", "Customer service") sont correctement identifies sans aucun
entrainement dedie, juste par grammaire (POS tagging). Cette baseline
sert de premiere etape avant la classification proprement dite.

La TOKENISATION EN PAIRE (texte, aspect) suit la convention standard
des Transformers : tokenizer(texte, aspect) insere automatiquement
[CLS] texte [SEP] aspect [SEP], avec des token_type_ids distinguant
les deux segments -- le modele apprend a lire "sachant que je me
concentre sur CET aspect precis, quel est le sentiment ?"

Necessite un acces reseau a huggingface.co (meme situation que
DistilBERT en Phase 6, XLM-R en Phase 8) -- bloque dans le sandbox
utilise pour ecrire ce code.
"""

from __future__ import annotations

from transformers_arch.fine_tuning import (
    fine_tune_model,
)

ASPECT_CATEGORIES = [
    "price",
    "delivery",
    "customer_service",
    "packaging",
    "product_quality",
    "website",
    "refund",
    "shipping",
    "warranty",
]


def load_semeval_absa(
    dataset_name: str = "tomaarsen/setfit-absa-semeval-restaurants",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Charge un vrai dataset ABSA (SemEval), le convertit en triples
    (texte, aspect, label), et le decoupe en train/eval -- UN SEUL
    appel, UN SEUL chargement, toujours le meme comportement (pas de
    logique conditionnelle selon les splits disponibles sur le Hub).

    Remplace le jeu jouet de 10 phrases repetees (voir
    absa_experiments.txt) qui, verifie empiriquement, menait le modele
    a un raccourci (mot d'aspect -> etiquette fixe) plutot qu'a une
    vraie lecture du contexte.

    Retourne (train_textes, train_aspects, train_labels,
    eval_textes, eval_aspects, eval_labels)."""
    from datasets import load_dataset
    from sklearn.model_selection import train_test_split

    ds = load_dataset(dataset_name, split="train")
    print(f"Colonnes : {ds.column_names}  |  {len(ds)} lignes brutes")

    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    texts, aspects, labels = [], [], []

    for row in ds:
        text = row.get("text")
        aspect = row.get("span") or row.get("aspect")
        raw_label = row.get("label")
        label = (
            label_map.get(raw_label.lower())
            if isinstance(raw_label, str)
            else raw_label
        )
        if text and aspect and label is not None:
            texts.append(text)
            aspects.append(aspect)
            labels.append(label)

    print(f"{len(texts)} exemples valides extraits")

    idx = list(range(len(texts)))
    idx_train, idx_eval = train_test_split(
        idx, test_size=test_size, random_state=random_state
    )

    def _select(items, indices):
        return [items[i] for i in indices]

    return (
        _select(texts, idx_train),
        _select(aspects, idx_train),
        _select(labels, idx_train),
        _select(texts, idx_eval),
        _select(aspects, idx_eval),
        _select(labels, idx_eval),
    )


def extract_aspect_candidates(text: str) -> list[str]:
    """Extrait des aspects candidats via la baseline grammaticale de
    la Phase 1 (groupes nominaux consecutifs). Premiere etape de
    l'architecture pipeline -- a comparer avec la performance de
    l'architecture jointe (qui n'a pas besoin de cette etape separee)."""
    from nltk import word_tokenize

    from preprocessing.linguistic import extract_noun_phrases

    tokens = word_tokenize(text)
    return extract_noun_phrases(tokens)


class AspectPairDataset:
    """Wrapper au format attendu par le Trainer de Hugging Face pour
    des paires (texte, aspect) -- meme principe que SentimentDataset
    de la Phase 6, mais chaque exemple encode DEUX segments."""

    def __init__(self, encodings, labels: list[int]):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> dict:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self) -> int:
        return len(self.labels)


def tokenize_aspect_pairs(
    texts: list[str],
    aspects: list[str],
    tokenizer,
    max_length: int = 256,
):
    """Tokenise des paires (texte, aspect) -- le tokenizer insere
    automatiquement [SEP] entre les deux segments et produit des
    token_type_ids qui les distinguent. texts et aspects doivent
    avoir la meme longueur (un aspect par texte)."""
    return tokenizer(
        texts,
        aspects,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )


def load_absa_classifier(num_labels: int = 3):
    """Charge DistilBERT avec une tete de classification pour l'ABSA
    -- 3 classes par defaut (positive/negative/neutral), contre 2 pour
    le sentiment global de la Phase 6 (pas de neutre)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    id2label = {0: "negative", 1: "neutral", 2: "positive"}
    label2id = {"negative": 0, "neutral": 1, "positive": 2}

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    return model, tokenizer


def train_absa_model(
    model,
    tokenizer,
    train_texts: list[str],
    train_aspects: list[str],
    train_labels: list[int],
    eval_texts: list[str],
    eval_aspects: list[str],
    eval_labels: list[int],
    epochs: int = 3,
):
    """Fine-tune le modele ABSA sur des paires (texte, aspect, label).
    Reutilise fine_tune_model() de la Phase 6 -- seule la tokenisation
    en paire change par rapport a un fine-tuning de sentiment global."""
    train_encodings = tokenize_aspect_pairs(train_texts, train_aspects, tokenizer)
    eval_encodings = tokenize_aspect_pairs(eval_texts, eval_aspects, tokenizer)

    train_dataset = AspectPairDataset(train_encodings, train_labels)
    eval_dataset = AspectPairDataset(eval_encodings, eval_labels)

    return fine_tune_model(model, train_dataset, eval_dataset, epochs=epochs)


def predict_aspect_sentiment(
    text: str, aspects: list[str], model, tokenizer
) -> dict[str, str]:
    """Pipeline complet pour UN avis : pour chaque aspect candidat
    fourni, predit son sentiment individuellement -- c'est cette
    fonction qui produit la sortie finale attendue par le projet :
    {aspect: sentiment} plutot qu'un score de sentiment global unique."""
    import torch

    results = {}
    for aspect in aspects:
        inputs = tokenize_aspect_pairs([text], [aspect], tokenizer)
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_class = torch.argmax(logits, dim=-1).item()
        results[aspect] = model.config.id2label[predicted_class]
    return results
