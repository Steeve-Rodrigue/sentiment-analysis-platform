"""
src/llms/prompting.py

Phase 7 -- LLMs -- bloc "prompting, in-context learning,
chain-of-thought".

Theorie resumee (voir la conversation associee pour le detail) :

Changement de paradigme par rapport au fine-tuning (Phase 6) : AUCUN
poids du modele n'est modifie. Toute l'intelligence reside dans la
FORMULATION de la question.

- ZERO-SHOT : la tache est demandee directement, sans exemple. Le
  modele s'appuie uniquement sur son pre-entrainement massif.
- FEW-SHOT : 2-3 exemples de la tache sont donnes DANS le prompt
  lui-meme, avant la vraie question -- le modele "apprend" le format
  et la tache a la volee, en lisant le prompt (in-context learning),
  sans jamais ajuster un seul poids.
- CHAIN-OF-THOUGHT : le modele est invite a expliciter son
  raisonnement etape par etape AVANT de conclure -- ameliore souvent
  la justesse sur des taches qui combinent plusieurs sous-problemes
  (ex. ABSA : identifier les aspects, PUIS le sentiment de chacun).

Les fonctions de CONSTRUCTION de prompts (ci-dessous) sont du texte
pur, testables sans aucun modele ni reseau. L'APPEL a un vrai LLM
(classify_with_llm) necessite en revanche un acces reseau a
huggingface.co, bloque dans le sandbox utilise pour ecrire ce code --
meme situation que le fine-tuning DistilBERT en Phase 6.
"""

from __future__ import annotations


def build_zero_shot_prompt(review: str, labels: tuple[str, ...] = None):
    """Construit un prompt zero-shot : demande directe, sans exemple."""
    labels = labels or ("positive", "negative", "mixed")
    labels_str = ", ".join(labels[:-1]) + f", or {labels[-1]}"
    return (
        f"Classify the sentiment of this review as {labels_str}:\n"
        f'"{review}"\nSentiment:'
    )


def build_few_shot_prompt(review: str, examples: list[tuple[str, str]]) -> str:
    """Construit un prompt few-shot : quelques exemples (review,
    sentiment) precedent la vraie question, pour montrer le format et
    la tache attendus sans toucher aux poids du modele."""
    lines = [f'Review: "{r}" -> Sentiment: {s}' for r, s in examples]
    lines.append(f'Review: "{review}" -> Sentiment:')
    return "\n".join(lines)


def build_chain_of_thought_prompt(review: str) -> str:
    """Construit un prompt chain-of-thought : demande au modele de
    raisonner explicitement, aspect par aspect, avant de conclure --
    pertinent pour l'ABSA (plusieurs aspects possibles dans un avis)."""
    return (
        f'Review: "{review}"\n'
        "Let's think step by step:\n"
        "1. Identify the aspects mentioned in this review.\n"
        "2. Determine the sentiment for each aspect separately.\n"
        "3. Combine into an overall sentiment (positive, negative, "
        "or mixed).\n"
        "Answer:"
    )


def classify_with_llm(prompt: str, model_name: str = "google/flan-t5-base") -> str:
    """Envoie un prompt a un vrai LLM generatif et retourne sa
    reponse. Necessite un acces reseau a huggingface.co pour
    telecharger le modele (bloque dans le sandbox, a lancer chez
    vous -- voir le fichier d'experimentation pour la verification
    complete).

    Utilise AutoModelForSeq2SeqLM + generate() directement plutot que
    pipeline("text2text-generation", ...) -- bug reel rencontre en
    testant : certaines versions de transformers n'exposent pas ce
    nom de tache dans le registre des pipelines
    (KeyError/ValueError "Unknown task"). Appeler le modele
    directement evite cette dependance fragile au nom exact de la
    tache."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
