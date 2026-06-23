"""
evaluate.py
Loads a trained model from HuggingFace Hub and runs evaluation.
Produces a benchmark report comparing full fine-tuning vs LoRA.
"""

import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


def load_scorer(repo_name: str):
    """Load a trained scorer from HuggingFace Hub."""
    tokenizer = AutoTokenizer.from_pretrained(repo_name)
    model = AutoModelForSequenceClassification.from_pretrained(repo_name)
    return tokenizer, model


def predict_score(resume: str, jd: str, tokenizer, model) -> float:
    """
    Run inference on a single resume-JD pair.
    Returns a score from 0-100.
    """
    inputs = tokenizer(
        resume,
        jd,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    import torch
    with torch.no_grad():
        output = model(**inputs)
    score = output.logits.squeeze().item() * 100
    return round(float(np.clip(score, 0, 100)), 2)


def evaluate_on_dataframe(df: pd.DataFrame, tokenizer, model) -> dict:
    """
    Run evaluation on a DataFrame with columns: resume, jd, score.
    Returns MAE, RMSE, and a results DataFrame.
    """
    predictions = []
    for _, row in df.iterrows():
        pred = predict_score(row["resume"], row["jd"], tokenizer, model)
        predictions.append(pred)

    df = df.copy()
    df["predicted"] = predictions
    df["error"] = df["predicted"] - df["score"]
    df["abs_error"] = df["error"].abs()

    mae = round(df["abs_error"].mean(), 4)
    rmse = round(np.sqrt((df["error"] ** 2).mean()), 4)

    return {
        "mae": mae,
        "rmse": rmse,
        "results_df": df,
    }


def print_benchmark_report(full_metrics: dict, lora_metrics: dict):
    """Print a side-by-side benchmark table."""
    print("\n===== BENCHMARK REPORT =====")
    print(f"{'Metric':<15} {'Full Fine-tune':>15} {'LoRA':>15}")
    print("-" * 47)
    print(f"{'MAE':<15} {full_metrics['mae']:>15} {lora_metrics['mae']:>15}")
    print(f"{'RMSE':<15} {full_metrics['rmse']:>15} {lora_metrics['rmse']:>15}")
    print("=" * 47)
