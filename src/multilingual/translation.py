"""
src/multilingual/translation_based.py

Phase 8 -- Multilingue -- bloc "traduction puis classification".

Theorie resumee (voir la conversation associee pour le detail) :

Approche la plus simple des 3 : traduire le texte vers l'anglais avec
un modele de traduction dedie (MarianMT, Helsinki-NLP), puis reutiliser
le modele DistilBERT anglais deja fine-tune en Phase 6 -- aucun
nouveau modele de classification a entrainer.

    sentiment = DistilBERT_EN(Traduction(texte_langue_source))

Avantage : reutilise tout le travail deja fait (Phase 6). Risque
principal : chaque erreur de traduction se repercute directement sur
la classification -- un contresens peut inverser completement le
sentiment percu, ce que ni XLM-R (Phase 8, zero-shot) ni un modele
multilingue fine-tune n'ont a subir (ils raisonnent DANS la langue
source, sans etape de traduction intermediaire qui pourrait deformer
le sens).

Necessite un acces reseau a huggingface.co pour telecharger les
modeles de traduction -- bloque dans le sandbox utilise pour ecrire ce
code, a verifier chez vous.
"""

from __future__ import annotations

# Modeles MarianMT (Helsinki-NLP) pour chaque langue source du projet
# vers l'anglais -- convention de nommage : Helsinki-NLP/opus-mt-{src}-en
TRANSLATION_MODELS = {
    "es": "Helsinki-NLP/opus-mt-es-en",
    "de": "Helsinki-NLP/opus-mt-de-en",
    "fr": "Helsinki-NLP/opus-mt-fr-en",
    "hi": "Helsinki-NLP/opus-mt-hi-en",
}

_TRANSLATOR_CACHE: dict = {}


def translate_to_english(text: str, source_lang: str) -> str:
    """Traduit un texte vers l'anglais avec le modele MarianMT dedie a
    la langue source. Met le modele/tokenizer en cache pour eviter de
    recharger a chaque appel (meme principe que glove_model.py,
    Phase 2).

    Utilise AutoModelForSeq2SeqLM + generate() directement plutot que
    pipeline("translation", ...) -- meme bug reel que classify_with_llm
    en Phase 7 : certaines versions de transformers n'exposent pas ce
    nom de tache dans le registre des pipelines ("Unknown task
    translation"). Appeler le modele directement evite cette
    dependance fragile au nom exact de la tache."""
    if source_lang not in TRANSLATION_MODELS:
        raise ValueError(
            f"Langue '{source_lang}' non supportee. "
            f"Choix : {list(TRANSLATION_MODELS)}"
        )

    if source_lang not in _TRANSLATOR_CACHE:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        model_name = TRANSLATION_MODELS[source_lang]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        _TRANSLATOR_CACHE[source_lang] = (model, tokenizer)

    model, tokenizer = _TRANSLATOR_CACHE[source_lang]
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def classify_via_translation(
    text: str, source_lang: str, classifier_model, classifier_tokenizer
) -> dict:
    """Pipeline complet : traduit vers l'anglais, puis classifie avec
    le modele DistilBERT anglais deja fine-tune (Phase 6). Retourne la
    traduction ET la prediction, pour pouvoir inspecter ou l'etape a
    pu se tromper en cas de resultat surprenant."""
    import torch

    translated = translate_to_english(text, source_lang)

    inputs = classifier_tokenizer(translated, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = classifier_model(**inputs).logits
    predicted_class = torch.argmax(logits, dim=-1).item()
    label = classifier_model.config.id2label[predicted_class]

    return {
        "original_text": text,
        "translated_text": translated,
        "predicted_label": label,
    }
