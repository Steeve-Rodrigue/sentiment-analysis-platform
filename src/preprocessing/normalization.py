"""
src/preprocessing/normalization.py

Phase 1 — NLP Fundamentals — bloc "normalisation".

Couvre : stemming, lemmatisation, suppression des stop-words.

Théorie résumée (voir la conversation associée pour le détail) :
- Le stemming coupe des suffixes par règles mécaniques -- rapide mais
  peut produire des non-mots ("university" -> "univers") ou fusionner
  à tort deux mots différents.
- La lemmatisation ramène un mot à sa forme de dictionnaire réelle, en
  s'appuyant sur sa catégorie grammaticale (POS) -- plus lente mais
  linguistiquement correcte ("delivered" -> "deliver", "better" -> "good"
  si on force pos="a").
- La suppression des stop-words est utile pour des modèles Bag-of-Words
  classiques (Phase 4), mais DANGEREUSE pour l'analyse de sentiment si
  elle retire des mots de négation ("not") -- voir negation.py, qui doit
  protéger ces mots avant tout appel à remove_stopwords().
"""

from __future__ import annotations

from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer

_STEMMERS = {
    "porter": PorterStemmer(),
    "snowball_en": SnowballStemmer("english"),
    "snowball_es": SnowballStemmer("spanish"),
    "snowball_de": SnowballStemmer("german"),
    "snowball_fr": SnowballStemmer("french"),
}
_LEMMATIZER = WordNetLemmatizer()

_STOPWORDS_CACHE: dict[str, set[str]] = {}


def get_stopwords(language: str = "english") -> set[str]:
    """Charge (et met en cache) la liste de stop-words NLTK pour une langue."""
    if language not in _STOPWORDS_CACHE:
        _STOPWORDS_CACHE[language] = set(stopwords.words(language))
    return _STOPWORDS_CACHE[language]


def remove_stopwords(tokens: list[str], language: str = "english") -> list[str]:
    """Retire les stop-words. ATTENTION : peut retirer des mots de
    négation ("not", "no") -- à combiner avec negation.py avant appel
    si le texte sert à de l'analyse de sentiment."""
    sw = get_stopwords(language)
    return [tok for tok in tokens if tok.lower() not in sw]


def stem_tokens(tokens: list[str], stemmer: str = "porter") -> list[str]:
    """Applique un stemmer mécanique. "porter" = anglais uniquement ;
    utiliser "snowball_<langue>" pour les autres langues du projet."""
    engine = _STEMMERS[stemmer]
    return [engine.stem(tok) for tok in tokens]


def _penn_to_wordnet_pos(penn_tag: str) -> str:
    """WordNet attend ses propres tags grossiers (n/v/a/r), pas les tags
    fins de Penn Treebank produits par le tagger NLTK par défaut."""
    if penn_tag.startswith("J"):
        return "a"  # adjectif
    if penn_tag.startswith("V"):
        return "v"  # verbe
    if penn_tag.startswith("R"):
        return "r"  # adverbe
    return "n"  # nom, par défaut


def lemmatize_tokens(tokens: list[str], use_pos: bool = True) -> list[str]:
    """Lemmatise en tenant compte de la POS par défaut -- sans ça,
    WordNet suppose que chaque mot est un nom, donc "running" (verbe)
    resterait "running" au lieu de devenir "run"."""
    if not use_pos:
        return [_LEMMATIZER.lemmatize(tok) for tok in tokens]
    tagged = pos_tag(tokens)
    return [
        _LEMMATIZER.lemmatize(tok, pos=_penn_to_wordnet_pos(tag)) for tok, tag in tagged
    ]
