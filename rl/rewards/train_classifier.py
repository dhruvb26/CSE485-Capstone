"""Fine-tune Flan-T5-small as a multi-label strategy classifier on CaSiNo
strategy annotations.

Usage:
    python -m rl.rewards.train_classifier \\
        --data data/casino/ca.train.csv \\
        --out checkpoints/strategy-classifier \\
        --epochs 10
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from rl.handlers.casino.dataset import (
    STRATEGY_LABEL_MAP,
    STRATEGY_LABELS,
    CasinoDatasetHandler,
    _META_TURNS,
    sanitize_unicode,
)

logger = logging.getLogger(__name__)

LABEL_LIST = sorted(STRATEGY_LABELS)
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
NUM_LABELS = len(LABEL_LIST)
BASE_MODEL = "google/flan-t5-small"


def _build_dataset(data_path: str) -> list[dict]:
    """Extract (utterance, multi-hot label vector) pairs from CaSiNo train data."""
    handler = CasinoDatasetHandler(data_path)
    instances = handler.get_instances()

    examples = []
    for inst in instances:
        chat_logs = inst["chat_logs"]
        annotations = inst["annotations"]

        for log_turn, ann in zip(chat_logs, annotations):
            if log_turn["text"] in _META_TURNS:
                continue
            raw_labels = ann[1] if len(ann) > 1 else ""
            if "non-strategic" in raw_labels:
                continue

            labels_set = set()
            for s in raw_labels.split(","):
                normed = STRATEGY_LABEL_MAP.get(s.strip(), s.strip())
                if normed in STRATEGY_LABELS:
                    labels_set.add(normed)

            if not labels_set:
                continue

            multi_hot = [1.0 if LABEL_LIST[i] in labels_set else 0.0 for i in range(NUM_LABELS)]
            text = sanitize_unicode(log_turn["text"])
            examples.append({"text": text, "labels": multi_hot})

    return examples


def _tokenize(examples: dict, tokenizer) -> dict:
    enc = tokenizer(
        examples["text"],
        truncation=True,
        max_length=256,
        padding="max_length",
    )
    enc["labels"] = examples["labels"]
    return enc


def train(data_path: str, out_dir: str, epochs: int, batch_size: int, lr: float) -> None:
    logger.info("Building dataset from %s", data_path)
    raw = _build_dataset(data_path)
    logger.info("Extracted %d strategy-labelled utterances", len(raw))

    if not raw:
        raise ValueError("No labelled utterances found.")

    dataset = Dataset.from_list(raw)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    dataset = dataset.map(
        lambda ex: _tokenize(ex, tokenizer),
        batched=True,
        remove_columns=["text"],
    )
    dataset.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
    )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_path),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    model.save_pretrained(str(out_path))
    tokenizer.save_pretrained(str(out_path))
    logger.info("Saved strategy classifier to %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Train CaSiNo strategy classifier")
    parser.add_argument("--data", required=True, help="Path to ca.train.csv")
    parser.add_argument("--out", required=True, help="Output directory for checkpoint")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    train(args.data, args.out, args.epochs, args.batch_size, args.lr)
