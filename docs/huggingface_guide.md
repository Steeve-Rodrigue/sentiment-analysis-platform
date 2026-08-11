# Guide Hugging Face
(https://huggingface.co/docs/huggingface_hub/en/guides/cli)

Ce guide couvre la configuration du compte et donne le code prêt à
l'emploi pour les 3 moments de publication du projet : modèle,
dataset, Space.

## 1. Compte et authentification

1. Créer un compte : https://huggingface.co/join
2. Créer un token d'accès **write** :
   https://huggingface.co/settings/tokens
3. Se connecter (une seule fois, le token reste en cache localement) :
   ```bash
   uv run huggingface-cli login --token TON_TOKEN
   ```

## 2. Publier un modèle fine-tuné (Phase 6, 7, 8, 9)

```python
from huggingface_hub import create_repo
from transformers import AutoModelForSequenceClassification, AutoTokenizer

repo_id = "TON_PSEUDO/globatrend-sentiment-distilbert"

create_repo(repo_id, exist_ok=True, repo_type="model")

model = AutoModelForSequenceClassification.from_pretrained(
    "./distilbert-sentiment"  # le output_dir de fine_tune_model()
)
tokenizer = AutoTokenizer.from_pretrained("./distilbert-sentiment")

model.push_to_hub(repo_id)
tokenizer.push_to_hub(repo_id)
```

Ou directement, comme dans `fine_tuning.py` :
```python
from transformer_arch.fine_tuning import push_to_hub

push_to_hub(model, tokenizer, "TON_PSEUDO/globatrend-sentiment-distilbert")
```

Ajoute ensuite une **Model Card** (le `README.md` du dépôt modèle sur
le Hub) — voir `docs/model_card_template.md`. Au minimum : données
d'entraînement, métriques, usage prévu, limites connues.

## 3. Publier un dataset (Phase 0 / Phase 9)

```python
from datasets import Dataset, DatasetDict

train_ds = Dataset.from_dict({"text": X_train, "label": y_train})
test_ds = Dataset.from_dict({"text": X_test, "label": y_test})

dataset = DatasetDict({"train": train_ds, "test": test_ds})
dataset.push_to_hub("TON_PSEUDO/globatrend-absa-unified")
```

Accompagne-le d'une **Datasheet** (`docs/datasheet_template.md`).

## 4. Déployer une démo (Space, Phase 14)

```python
# app/gradio_app.py
import gradio as gr
from transformers import pipeline

classifieur = pipeline(
    "text-classification",
    model="TON_PSEUDO/globatrend-sentiment-distilbert",
)

def analyser(avis):
    return classifieur(avis)

demo = gr.Interface(
    fn=analyser,
    inputs=gr.Textbox(lines=4, placeholder="Colle un avis client..."),
    outputs="json",
    title="GlobaTrend Insights",
)
demo.launch()
```

```bash
uv run huggingface-cli repo create globatrend-insights-demo \
    --type space --space_sdk gradio
git clone https://huggingface.co/spaces/TON_PSEUDO/globatrend-insights-demo
cp app/gradio_app.py globatrend-insights-demo/app.py
uv export --no-hashes --format requirements-txt \
    > globatrend-insights-demo/requirements.txt
cd globatrend-insights-demo && git add . && git commit -m "demo" && git push
```

## 5. Noms de dépôts suggérés

| Artefact | Nom |
|---|---|
| Modèle sentiment (Phase 6) | `globatrend-sentiment-distilbert` |
| Modèle multilingue (Phase 8) | `globatrend-xlm-r-multilingual-sentiment` |
| Modèle ABSA phare (Phase 9) | `globatrend-absa-classifier` |
| Dataset unifié | `globatrend-absa-unified` |
| Space de démo | `globatrend-insights-demo` |
