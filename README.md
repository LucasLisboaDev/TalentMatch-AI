# TalentMatch.ai — Fine-tuned DistilBERT for Resume-JD Scoring

TalentMatch.ai is an NLP model that quantifies resume-to-job-description fit using a continuous 0-100 match score. Fine-tuned on `distilbert-base-uncased` using both full fine-tuning and LoRA (parameter-efficient fine-tuning), it covers the full ML lifecycle — dataset preprocessing, model training, evaluation benchmarking, and deployment to HuggingFace Hub with a live Gradio demo. Built to demonstrate production-grade ML engineering practices on constrained compute (Google Colab free tier).

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
├── data/                             # Gitignored — data lives on HuggingFace Datasets
└── requirements.txt
```

---

## Model

- **Base model:** `distilbert-base-uncased`
- **Task:** Regression (continuous score 0-100)
- **Dataset:** `netsol/resume-score-details` (1,031 resume-JD pairs, GPT-4o labeled)
- **Fine-tuning approaches:** Full fine-tuning vs LoRA via `peft`

---

## Benchmark Results

| Approach | MAE | RMSE | Training Time | GPU Memory |
|---|---|---|---|---|
| Full fine-tuning | TBD | TBD | TBD | TBD |
| LoRA fine-tuning | TBD | TBD | TBD | TBD |

*(Updated after training runs)*

---

## Quickstart

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("LucasLisboaDev/TalentMatch-AI")
model = AutoModelForSequenceClassification.from_pretrained("LucasLisboaDev/TalentMatch-AI")

resume = "Python engineer with 3 years experience in FastAPI, Docker, and PostgreSQL."
jd = "Backend Engineer — requires Python, REST APIs, and cloud deployment experience."

inputs = tokenizer(resume, jd, return_tensors="pt", truncation=True, max_length=512)
with torch.no_grad():
    score = model(**inputs).logits.squeeze().item() * 100

print(f"Match Score: {round(np.clip(score, 0, 100), 2)}/100")
```

---

## Setup

```bash
pip install -r requirements.txt
```

To run training notebooks, open them directly in Google Colab for free GPU access:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LucasLisboaDev/TalentMatch-AI/blob/main/notebooks/01_data_exploration.ipynb)

---

## Stack

- HuggingFace `transformers` + `datasets` + `evaluate`
- `peft` for LoRA fine-tuning
- `accelerate` for training optimization
- Gradio for demo UI
- Google Colab (free tier) for GPU training
- HuggingFace Hub for model hosting

---

## Author

Lucas Lisboa — [GitHub](https://github.com/LucasLisboaDev) | [Portfolio](https://portfolio-lucas-henna.vercel.app)
