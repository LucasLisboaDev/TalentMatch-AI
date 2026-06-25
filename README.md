# TalentMatch.ai — Fine-tuned DistilBERT for Resume-JD Scoring

TalentMatch.ai is an NLP model that quantifies resume-to-job-description fit using a continuous 0-100 match score. Fine-tuned on `distilbert-base-uncased` using both full fine-tuning and LoRA (parameter-efficient fine-tuning), it covers the full ML lifecycle — dataset preprocessing, model training, evaluation benchmarking, and deployment to HuggingFace Hub. Built to demonstrate production-grade ML engineering practices on constrained compute (Google Colab free tier).

---

## Key Finding

> Full fine-tuning outperforms LoRA on small datasets with small models. LoRA is designed for large models (7B+ params) where full fine-tuning is computationally infeasible. On DistilBERT (67M params) with 680 training samples, full fine-tuning achieved **32% lower MAE** than LoRA while LoRA's adapter weighed **90x less** (3MB vs 268MB).

---

## Benchmark Results

| Metric | Baseline | Full Fine-Tuning | LoRA |
|---|---|---|---|
| Test MAE (↓ better) | 17.69 | **11.95** | 17.51 |
| Test RMSE (↓ better) | 21.23 | **14.90** | 21.12 |
| Trainable Parameters | N/A | 66.9M (100%) | 0.74M (1.09%) |
| Training Time (T4 GPU) | N/A | 90s | 54s |
| HuggingFace Hub Size | N/A | 268MB | 3MB |
| HuggingFace Hub | N/A | [TalentMatch-AI-full](https://huggingface.co/LucasLisboadev/TalentMatch-AI-full) | [TalentMatch-AI-lora](https://huggingface.co/LucasLisboadev/TalentMatch-AI-lora) |

**Baseline:** always predicting the mean training score (56.4).  
**MAE improvement over baseline:** Full FT = 32% | LoRA = 1%

---

## Project Structure

```
TalentMatch-AI/
├── notebooks/
│   ├── 01_data_exploration.ipynb     # Dataset loading, schema inspection, score normalization
│   ├── 02_full_finetune.ipynb        # Full fine-tuning with HuggingFace Trainer API
│   └── 03_lora_finetune.ipynb        # Parameter-efficient fine-tuning with LoRA (PEFT)
├── src/
│   ├── dataset.py                    # Dataset loading and preprocessing pipeline
│   ├── train.py                      # Training logic, Trainer config, push to Hub
│   └── evaluate.py                   # Evaluation metrics, benchmark report
├── data/                             # Gitignored — processed splits live on Google Drive
└── requirements.txt
```

---

## Dataset

- **Source:** `netsol/resume-score-details` — 1,031 resume-JD pairs labeled by GPT-4o
- **After cleaning:** 851 valid samples (180 dropped — invalid flags, missing scores, gibberish inputs)
- **Split:** 680 train / 85 validation / 86 test
- **Score normalization:** GPT-4o rubric scores (0-10 scale) combined with 60/40 macro/micro weighting and rescaled to 0-100
- **Distribution:** Mean 55.2, Std 19.3, range 0-98.3 — roughly bell-shaped, no dominant class

---

## Model

- **Base model:** `distilbert-base-uncased`
- **Task:** Regression — single continuous output (match score 0-100)
- **Loss function:** MSELoss (automatically selected with `num_labels=1`)
- **Input format:** `[CLS] resume [SEP] job_description [SEP]` — max 512 tokens
- **Label normalization:** scores divided by 100 during training for gradient stability, rescaled for evaluation

---

## Quickstart

```python
# Full fine-tuned model
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, numpy as np

tokenizer = AutoTokenizer.from_pretrained("LucasLisboadev/TalentMatch-AI-full")
model = AutoModelForSequenceClassification.from_pretrained("LucasLisboadev/TalentMatch-AI-full")
model.eval()

resume = "Senior Python engineer with 5 years building FastAPI backends and PostgreSQL databases."
jd = "Backend Engineer: Python, FastAPI, PostgreSQL. 3+ years experience required."

inputs = tokenizer(resume, jd, return_tensors="pt", truncation=True, max_length=512)
with torch.no_grad():
    score = model(**inputs).logits.squeeze().item() * 100

print(f"Match Score: {round(np.clip(score, 0, 100), 2)}/100")
```

```python
# LoRA adapter model
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer  = AutoTokenizer.from_pretrained("distilbert-base-uncased")
base_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=1)
model      = PeftModel.from_pretrained(base_model, "LucasLisboadev/TalentMatch-AI-lora")
```

---

## Engineering Decisions

**Why dynamic max normalization instead of a fixed scale?**
The dataset documentation implied a 0-5 score scale but the actual range was 0-9.55. Hardcoding 5.0 would have silently clipped ~60% of labels to 100 — a silent label corruption bug. We computed the max dynamically from the data.

**Why 60/40 macro/micro weighting?**
Macro scores evaluate strategic fit (leadership, domain experience). Micro scores evaluate specific tool matches (HubSpot, SEO). In real hiring, overall fit matters more than keyword matching — the weighting encodes that judgment into the label.

**Why full fine-tuning outperformed LoRA here?**
LoRA excels on large models (7B+ params) where updating all weights is computationally infeasible. On DistilBERT (67M params) with only 680 training samples, freezing 99% of weights is too restrictive — the model can't adapt enough to the specific scoring task. Full fine-tuning had the freedom to adjust all layers and achieved 32% better MAE.

---

## Setup

```bash
pip install -r requirements.txt
```

Open notebooks in Google Colab for free T4 GPU access:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LucasLisboaDev/TalentMatch-AI/blob/main/notebooks/01_data_exploration.ipynb)

---

## Stack

- HuggingFace `transformers` + `datasets` + `evaluate`
- `peft` for LoRA fine-tuning
- `accelerate` for mixed precision training
- Google Colab T4 GPU for training
- Google Drive for intermediate data storage
- HuggingFace Hub for model hosting

---

## Author

Lucas Lisboa — [GitHub](https://github.com/LucasLisboaDev) | [Portfolio](https://portfolio-lucas-henna.vercel.app) | [HuggingFace](https://huggingface.co/LucasLisboadev)