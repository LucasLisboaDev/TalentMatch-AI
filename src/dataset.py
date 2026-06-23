"""
dataset.py
Handles loading the netsol/resume-score-details dataset from HuggingFace,
inspecting the schema, cleaning invalid samples, and normalizing scores to 0-100.
"""

from datasets import load_dataset
import pandas as pd
import numpy as np


def load_raw_dataset():
    """Load the raw dataset from HuggingFace Hub."""
    dataset = load_dataset("netsol/resume-score-details")
    return dataset


def extract_fields(sample: dict) -> dict:
    """
    Extract the fields we need from each sample.
    The dataset stores nested JSON — we flatten it here.
    Returns None if the sample is invalid.
    """
    try:
        output = sample.get("output", {})
        scores = output.get("scores", {})
        aggregated = scores.get("aggregated_scores", {})

        resume_text = sample.get("resume_text", "")
        jd_text = sample.get("jd_text", "")
        macro_score = aggregated.get("macro_scores", None)
        micro_score = aggregated.get("micro_scores", None)
        valid = output.get("valid_resume_and_jd", False)

        if not valid or not resume_text or not jd_text:
            return None
        if macro_score is None or micro_score is None:
            return None

        return {
            "resume": resume_text,
            "jd": jd_text,
            "macro_score": float(macro_score),
            "micro_score": float(micro_score),
        }
    except Exception:
        return None


def normalize_score(macro: float, micro: float, scale: float = 5.0) -> float:
    """
    Combine macro and micro scores and normalize to 0-100.
    Both scores are on a 0-scale (default 5.0).
    We weight macro slightly higher (60/40) as it reflects overall fit.
    """
    combined = (0.6 * macro + 0.4 * micro)
    normalized = (combined / scale) * 100
    return round(float(np.clip(normalized, 0, 100)), 2)


def build_dataframe(dataset) -> pd.DataFrame:
    """
    Process the raw dataset into a clean DataFrame with columns:
    - resume: str
    - jd: str
    - score: float (0-100)
    """
    records = []
    for sample in dataset["train"]:
        extracted = extract_fields(sample)
        if extracted is None:
            continue
        score = normalize_score(extracted["macro_score"], extracted["micro_score"])
        records.append({
            "resume": extracted["resume"],
            "jd": extracted["jd"],
            "score": score,
        })

    df = pd.DataFrame(records)
    return df


def get_score_distribution(df: pd.DataFrame) -> dict:
    """Return basic stats about the score distribution."""
    return {
        "count": len(df),
        "mean": round(df["score"].mean(), 2),
        "std": round(df["score"].std(), 2),
        "min": round(df["score"].min(), 2),
        "max": round(df["score"].max(), 2),
        "q25": round(df["score"].quantile(0.25), 2),
        "q50": round(df["score"].quantile(0.50), 2),
        "q75": round(df["score"].quantile(0.75), 2),
    }
