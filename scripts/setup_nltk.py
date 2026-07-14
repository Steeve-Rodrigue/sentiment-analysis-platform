"""
scripts/setup_nltk.py

One-time setup script: downloads all NLTK data packages this project
depends on. Run once after `uv sync`, and again by anyone reproducing
the environment (documented in README.md).

Why this exists as a separate script rather than being handled by
`uv sync`: NLTK data files (dictionaries, tagger models) are not Python
packages -- they are downloaded by NLTK itself into a local folder
(~/nltk_data by default), a mechanism entirely outside of uv's
dependency resolution. Without this explicit step, `uv sync` succeeds
but the code fails at runtime the first time it calls a function that
needs one of these resources.
"""

import nltk
from spacy.cli import download

REQUIRED_NLTK_PACKAGES = [
    "punkt_tab",  # sentence & word tokenization (Punkt model)
    "averaged_perceptron_tagger_eng",  # POS tagging
    "wordnet",  # lemmatization (WordNet dictionary)
    "stopwords",  # stop-word lists per language
    "maxent_ne_chunker_tab",  # named entity recognition
    "words",  # word lists
]

REQUIRED_SPACY_MODELS = [
    "en_core_web_sm",
    # "es_core_news_sm",  # uncomment when Phase 8/9 needs Spanish dependency parsing
    # "de_core_news_sm",  # uncomment when Phase 8/9 needs German dependency parsing
    # "fr_core_news_sm",  # uncomment when Phase 8/9 needs French dependency parsing
]


def main() -> None:
    for model_name in REQUIRED_SPACY_MODELS:
        print(f"Downloading spaCy model: {model_name}")
        download(model_name)
    print("\nAll required spaCy models downloaded successfully.")

    for package in REQUIRED_NLTK_PACKAGES:
        print(f"Downloading NLTK package: {package}")
        nltk.download(package)
    print("\nAll required NLTK data downloaded successfully.")


if __name__ == "__main__":
    main()
