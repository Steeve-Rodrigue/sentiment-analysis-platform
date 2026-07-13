# 🌍 GlobaTrend Insights — Multilingual Aspect-Based Sentiment Analysis Platform

> An end-to-end, portfolio-grade NLP platform built to **master NLP from first principles to LLMs**: classical ML → deep learning → transformers → aspect-based sentiment → real-time streaming → explainability → responsible AI → deployment.

[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-yellow)](#)
[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-blue)](#)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-orange)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![uv](https://img.shields.io/badge/managed%20with-uv-purple)](https://docs.astral.sh/uv/)

> 🔗 **Live demo:** _(Hugging Face Space link — added once the dashboard is deployed)_
> 🔗 **Fine-tuned models:** _(Hugging Face Hub link — added once the first model is trained)_
> 🔗 **Unified dataset:** _(Hugging Face Datasets link — added once the preprocessing pipeline is published)_

---

## 🧩 The fictional use case

**GlobaTrend Insights** is a fictional sentiment-analytics vendor for global e-commerce clients. The platform ingests customer reviews, support conversations, and social posts in **English, Spanish, German, Hindi, and French**, and answers a question a plain sentiment score can't: *the customer is happy or unhappy — but about **what**, exactly* (price, delivery, packaging, customer service, product quality, website, refund, shipping, warranty)?

## ❓ Problématique

> **Comment construire un système d'analyse de sentiment par aspect qui reste fiable et équitable quand on passe d'une langue à forte ressource (anglais) à des langues à ressources plus limitées (hindi notamment), sans devoir entraîner un modèle dédié par langue et par aspect ?**

This question drives the technical choices across the project — not just "cover every NLP concept," but **measure**, with real numbers on this project's own data, how much performance and fairness degrade as we move from English to lower-resource languages, and which cross-lingual strategy degrades the least.

## 🎯 Why this project exists

1. **Mastery** — cover the full modern NLP stack, from regex cleaning to LLM prompting, with working code for every concept — not just imported libraries. Each module explains: why it exists → theory & math → implementation → evaluation → limitations → industrial use case.
2. **Portfolio** — every trained model, dataset, and dashboard is published on **Hugging Face Hub**, so the work is verifiable and usable by anyone with one click, not buried in a local notebook.

---

## 🗺️ Roadmap (14 phases)

Full detail (theory notes, datasets, HF publication plan per phase) lives in [`ROADMAP.md`](ROADMAP.md). Status: ✅ done · 🚧 in progress · ⬜ planned.

| # | Phase | Status |
|---|---|---|
| 0 | Repository scaffolding, environment (uv), CI | 🚧 |
| 1 | NLP Fundamentals (cleaning, tokenization, linguistics) | ⬜ |
| 2 | Text Representation (BoW → Transformer embeddings) | ⬜ |
| 3 | Tokenization deep-dive (BPE, WordPiece, SentencePiece) | ⬜ |
| 4 | Classical Machine Learning | ⬜ |
| 5 | Deep Learning (CNN, RNN, LSTM, Attention) | ⬜ |
| 6 | Transformers (architecture + fine-tuning) | ⬜ |
| 7 | Large Language Models & prompting/RAG | ⬜ |
| 8 | Multilingual NLP (EN/ES/DE/HI/FR) | ⬜ |
| 9 | Aspect-Based Sentiment Analysis (core product) | ⬜ |
| 10 | Real-Time Processing (Kafka streaming) | ⬜ |
| 11 | Time-Series Analytics & dashboards | ⬜ |
| 12 | Explainable AI (SHAP, LIME, attention) | ⬜ |
| 13 | Responsible AI (fairness, bias, model cards) | ⬜ |
| 14 | Deployment (FastAPI, Docker, HF Spaces, CI/CD) | ⬜ |

**Progression rule:** never move to phase N+1 until phase N has (1) code that runs error-free on a real example, (2) a numerical evaluation, (3) a dedicated git commit, and (4) a Hugging Face publication if applicable. Better 5 phases genuinely mastered than a 14-phase facade.

---

## 📊 Datasets (verified, per language)

No single dataset covers all 5 languages with aspect-level annotation — that gap is precisely what makes cross-lingual transfer (Phase 8) necessary rather than cosmetic.

| Language | ABSA dataset (aspect-annotated) | Sentiment fallback |
|---|---|---|
| English | SemEval-2014 Task 4 (restaurant/laptop) | IMDb, Yelp |
| Spanish | SemEval-2016 Task 5 (restaurant, ES) | Multilingual Amazon Reviews Corpus (MARC) |
| French | SemEval-2016 Task 5 (restaurant, FR) | MARC |
| German | GERestaurant (Hellwig et al., 2024) | MARC |
| Hindi | `ai4bharat/IndicSentiment` (has `ASPECTS` columns) | — (MARC does **not** cover Hindi) |

A unified preprocessing pipeline (Phase 1/9) harmonizes these different annotation schemas (SemEval E#A categories vs. IndicSentiment columns vs. GERestaurant) before training a single multilingual model.

---

## 🤗 What lives on Hugging Face vs. GitHub

Hugging Face only hosts specific artifact types — the rest of the project (code, notebooks, tests, infra) lives here on GitHub.

| Artifact | Hosted on |
|---|---|
| Fine-tuned models (DistilBERT, XLM-R, ABSA classifier) + Model Cards | 🤗 Model Hub |
| Unified multilingual dataset + Datasheet | 🤗 Datasets Hub |
| Interactive dashboard (Streamlit) | 🤗 Space |
| Collection grouping all of the above | 🤗 Collections |
| All source code, notebooks, tests, Kafka/Docker infra, FastAPI, CI/CD, MLflow | GitHub (this repo) |

See [`docs/huggingface_guide.md`](docs/huggingface_guide.md) for account setup and ready-to-run push scripts.

---

## 📁 Repository structure

```
sentiment-analysis-platform/
├── data/
│   ├── raw/              # downloaded raw data (gitignored)
│   ├── processed/        # cleaned data ready for training
│   └── external/         # third-party resources (lexicons, pretrained embeddings)
├── notebooks/            # one teaching notebook per phase (01_..., 02_..., ...)
├── src/
│   ├── preprocessing/    # Phase 1
│   ├── embeddings/       # Phase 2
│   ├── tokenization/     # Phase 3
│   ├── classical_ml/     # Phase 4
│   ├── deep_learning/    # Phase 5
│   ├── transformers/     # Phase 6
│   ├── llms/             # Phase 7
│   ├── multilingual/     # Phase 8
│   ├── aspect_sentiment/ # Phase 9
│   ├── realtime/         # Phase 10
│   ├── dashboard/        # Phase 11
│   ├── explainability/   # Phase 12
│   ├── deployment/       # Phase 14
│   └── utils/            # shared helpers
├── configs/              # config.yaml (languages, aspects, paths, seed)
├── models/               # local artifacts (gitignored, synced to HF Hub)
├── reports/
│   └── figures/          # confusion matrices, ROC curves, benchmark charts
├── tests/                # pytest unit tests, mirrors src/
├── docker/               # Dockerfile, docker-compose (Phase 14)
├── app/                  # Streamlit/Gradio app (mirrors the HF Space)
├── api/                  # FastAPI service (Phase 14)
├── docs/
│   ├── huggingface_guide.md
│   ├── model_card_template.md
│   └── datasheet_template.md
├── scripts/
│   └── setup_nltk.py     # one-time NLTK data download, run after uv sync
├── .github/workflows/ci.yml
├── pyproject.toml        # dependencies (managed with uv)
├── uv.lock               # pinned, reproducible lockfile
├── .gitignore
├── LICENSE
├── ROADMAP.md
└── README.md
```

---

## 🚀 Getting started

This project uses **[uv](https://docs.astral.sh/uv/)** for environment and dependency management — no `pip`/`venv` needed.

```bash
git clone https://github.com/YOUR_USERNAME/sentiment-analysis-platform.git
cd sentiment-analysis-platform
uv sync
```

Some dependencies (NLTK) need one-time data downloads that `uv sync` does **not** cover, since these are data files fetched by NLTK itself, not Python packages tracked by uv. Run this once after `uv sync`:
```bash
uv run python scripts/setup_nltk.py
```

Run anything inside the managed environment with `uv run`:
```bash
uv run python src/preprocessing/pipeline.py
uv run jupyter lab
```

Add a new dependency as a phase needs it (don't front-load everything):
```bash
uv add nltk spacy langdetect emoji contractions   # e.g. for Phase 1
```

## 🧪 Testing

```bash
uv run pytest tests/ -v
```

## 📜 License

MIT — see [LICENSE](LICENSE).
