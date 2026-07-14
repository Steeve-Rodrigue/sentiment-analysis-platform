"""
src/preprocessing/linguistic.py

PMoteurs :
- NLTK pour POS tagging et NER -- fonctionne partout, aucun gros modèle
  à télécharger.
- spaCy pour le dependency parsing -- aucun équivalent NLTK suffisant
  pour nos langues ; nécessite un modèle téléchargé localement (voir
  scripts/verify_dependency_parsing.py).
"""

from __future__ import annotations

import spacy
from nltk import ne_chunk, pos_tag
from nltk.tree import Tree

_SPACY_MODELS: dict[str, "spacy.language.Language"] = {}

_SPACY_MODEL_NAMES = {
    "en": "en_core_web_sm",
    "es": "es_core_news_sm",
    "de": "de_core_news_sm",
    "fr": "fr_core_news_sm",
    # Pas de petit modèle spaCy officiel pour le hindi actuellement ;
    # les Phases 8/9 traitent le hindi via le transformer multilingue à la place.
}


def pos_tag_tokens(tokens: list[str]) -> list[tuple[str, str]]:
    """Retourne [(token, tag Penn-Treebank), ...],
    ex. [("delivery", "NN"), ("was", "VBD"), ("slow", "JJ")]."""
    return pos_tag(tokens)


def extract_noun_phrases(tokens: list[str]) -> list[str]:
    """Extracteur d'aspects candidats baseline : fusionne les tags noms
    (NN, NNS, NNP...) consécutifs. C'est une baseline volontaire, à
    comparer avec l'extracteur appris de la Phase 9."""
    tagged = pos_tag_tokens(tokens)
    phrases, current = [], []
    for word, tag in tagged:
        if tag.startswith("NN"):
            current.append(word)
        else:
            if current:
                phrases.append(" ".join(current))
                current = []
    if current:
        phrases.append(" ".join(current))
    return phrases


def named_entities_nltk(tokens: list[str]) -> list[tuple[str, str]]:
    """Retourne [(texte_entité, label), ...] via le chunker intégré à
    NLTK. Labels plus grossiers que spaCy (PERSON, ORGANIZATION, GPE),
    mais ne nécessite aucun téléchargement de modèle supplémentaire."""
    tagged = pos_tag_tokens(tokens)
    tree = ne_chunk(tagged)
    entities = []
    for subtree in tree:
        if isinstance(subtree, Tree):
            entity_text = " ".join(tok for tok, _ in subtree.leaves())
            entities.append((entity_text, subtree.label()))
    return entities


def _load_spacy_model(lang: str = "en"):
    if lang not in _SPACY_MODEL_NAMES:
        raise ValueError(f"Aucun modèle spaCy configuré pour la langue '{lang}'.")
    model_name = _SPACY_MODEL_NAMES[lang]
    if model_name not in _SPACY_MODELS:
        try:
            _SPACY_MODELS[model_name] = spacy.load(model_name)
        except OSError as exc:
            raise RuntimeError(
                f"Le modèle spaCy '{model_name}' n'est pas téléchargé localement.\n"
                f"Lancez une fois sur votre machine (accès internet requis) :\n"
                f"    uv run python -m spacy download {model_name}"
            ) from exc
    return _SPACY_MODELS[model_name]


def named_entities_spacy(text: str, lang: str = "en") -> list[tuple[str, str]]:
    """NER neuronal via spaCy -- plus précis que le chunker NLTK, mais
    nécessite le modèle téléchargé (voir message d'erreur sinon)."""
    nlp = _load_spacy_model(lang)
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]


def dependency_parse(text: str, lang: str = "en") -> list[dict]:
    """Retourne une liste de {token, dep, head, pos} décrivant la relation
    grammaticale de chaque token avec son head syntaxique. Nécessite
    spaCy + son pipeline entraîné -- aucun équivalent NLTK suffisant pour
    les langues de ce projet."""
    nlp = _load_spacy_model(lang)
    doc = nlp(text)
    return [
        {"token": tok.text, "dep": tok.dep_, "head": tok.head.text, "pos": tok.pos_}
        for tok in doc
    ]
