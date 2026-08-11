"""
src/transformer_arch/fine_tuning.py

Phase 6 -- Transformers -- bloc "fine-tuning DistilBERT".

PREMIER MODELE REEL DE TOUT LE PROJET A ETRE POUSSE SUR HUGGING FACE
(voir docs/huggingface_guide.md pour le code de publication).

Contrairement a tous les modeles de la Phase 5 (poids ALEATOIRES,
tout appris depuis zero sur 1600 exemples), DistilBERT a deja ete
entraine par Hugging Face sur des MILLIARDS de mots (Wikipedia +
BookCorpus) via masked language modeling. Le fine-tuning ne fait que
CONTINUER cet entrainement, avec un taux d'apprentissage tres faible
et peu d'epoques, pour adapter le modele a la tache de sentiment sans
detruire ce qu'il sait deja.

NOTE IMPORTANTE (verifiee empiriquement) : ce module necessite un
acces reseau a huggingface.co pour telecharger le modele -- bloque
dans l'environnement sandbox utilise pour ecrire ce code (meme erreur
que sentence-transformers et spaCy en Phases 2 et 3). Le code est
correct et pret a l'emploi, mais la VERIFICATION reelle (chiffres
observes) doit se faire sur votre machine -- voir le fichier
d'experimentation associe pour les instructions completes.

DistilBERT est une version COMPRESSEE de BERT (distillation de
connaissances) : ~40% de parametres en moins, ~60% plus rapide,
conserve ~97% de la performance de BERT sur la plupart des taches --
un bon compromis pour ce projet.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def compute_metrics(eval_pred) -> dict:
    """Calcule accuracy et F1 a partir des predictions brutes du
    Trainer. SANS cette fonction, Trainer.evaluate() ne retourne QUE
    la perte (eval_loss) -- bug reel rencontre en testant : la cle
    'eval_accuracy' etait absente du dict retourne, donnant nan."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions),
    }


def load_pretrained_classifier(
    model_name: str = "distilbert-base-uncased", num_labels: int = 2
):
    """Charge un tokenizer et un modele DistilBERT pre-entraine, avec
    une tete de classification NEUVE (poids aleatoires) ajoutee par
    dessus le corps pre-entraine -- seule cette tete part de zero,
    le reste du modele part deja "sachant lire"."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    return model, tokenizer


def tokenize_dataset(texts: list[str], tokenizer, max_length: int = 256):
    """Tokenise une liste de textes avec le tokenizer DistilBERT
    (WordPiece, Phase 3). truncation=True et padding=True gerent
    automatiquement les longueurs variables -- meme probleme que le
    padding manuel de la Phase 5, mais pris en charge nativement ici."""
    return tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )


class SentimentDataset:
    """Wrapper minimal au format attendu par le Trainer de Hugging
    Face : __getitem__ retourne un dict de tenseurs pour un exemple,
    __len__ le nombre total d'exemples."""

    def __init__(self, encodings, labels: list[int]):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> dict:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self) -> int:
        return len(self.labels)


def fine_tune_model(
    model,
    train_dataset,
    eval_dataset,
    output_dir: str = "./distilbert-sentiment",
    epochs: int = 2,
    learning_rate: float = 2e-5,
    batch_size: int = 16,
):
    """Fine-tune le modele avec le Trainer de Hugging Face.

    learning_rate volontairement TRES faible (2e-5, contre 0.001-0.01
    utilise en Phase 5) : le modele sait deja beaucoup de choses, un
    taux eleve risquerait d'ecraser ce savoir plutot que de l'affiner.
    epochs volontairement PEU nombreuses (2-3) : trop d'epoques sur un
    modele deja competent mene vite au surapprentissage."""
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    return trainer


def evaluate_fine_tuned_model(trainer) -> dict:
    """Evalue le modele fine-tune sur son eval_dataset."""
    return trainer.evaluate()


def push_to_hub(model, tokenizer, repo_id: str) -> None:
    """Publie le modele fine-tune sur le Hugging Face Hub. Necessite
    un token d'ecriture configure (uv run huggingface-cli login), voir
    docs/huggingface_guide.md pour la procedure complete."""
    model.push_to_hub(repo_id)
    tokenizer.push_to_hub(repo_id)
