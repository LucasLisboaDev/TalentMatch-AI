"""
train.py
Training logic for both full fine-tuning and LoRA fine-tuning.
Designed to be called from Colab notebooks.
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import Dataset
import numpy as np


MAX_LENGTH = 512
MODEL_NAME = "distilbert-base-uncased"
HF_REPO = "LucasLisboaDev/TalentMatch-AI"


class ResumeJDDataset(Dataset):
    """
    PyTorch Dataset that tokenizes resume+JD pairs.
    Input format: [CLS] resume [SEP] jd [SEP]
    Label: continuous score 0-100, normalized to 0-1 for training stability.
    """

    def __init__(self, df, tokenizer):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        encoding = self.tokenizer(
            row["resume"],
            row["jd"],
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            # Normalize label to 0-1 range for regression stability
            "labels": torch.tensor(row["score"] / 100.0, dtype=torch.float),
        }


def get_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def get_full_model():
    """Load DistilBERT with a regression head (num_labels=1)."""
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=1,  # regression, not classification
    )
    return model


def get_lora_model():
    """Wrap DistilBERT with LoRA adapters via PEFT."""
    base_model = get_full_model()
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,                          # rank — lower = fewer params
        lora_alpha=16,                # scaling factor
        lora_dropout=0.1,
        target_modules=["q_lin", "v_lin"],  # DistilBERT attention layers
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    return model


def get_training_args(output_dir: str, epochs: int = 3) -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=50,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        fp16=torch.cuda.is_available(),  # use mixed precision if GPU available
        report_to="none",
    )


def compute_metrics(eval_pred):
    """MAE and RMSE on the 0-100 scale."""
    predictions, labels = eval_pred
    predictions = predictions.flatten() * 100  # rescale back
    labels = labels * 100
    mae = np.mean(np.abs(predictions - labels))
    rmse = np.sqrt(np.mean((predictions - labels) ** 2))
    return {"mae": round(mae, 4), "rmse": round(rmse, 4)}


def train_and_push(model, tokenizer, train_dataset, eval_dataset, output_dir, repo_name):
    """Run training and push the best checkpoint to HuggingFace Hub."""
    args = get_training_args(output_dir)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.push_to_hub(repo_name)
    tokenizer.push_to_hub(repo_name)
    print(f"Model pushed to: https://huggingface.co/{repo_name}")
    return trainer
