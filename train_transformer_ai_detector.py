"""
Train a Transformer AI detector incrementally.

Features
--------
✓ Resume training
✓ Train in 10,000 sample chunks
✓ Apple Silicon (MPS) support
✓ Dynamic padding
✓ Automatic checkpoint loading
✓ Resume after interruption
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

LABEL2ID = {
    "human": 0,
    "ai": 1,
}

ID2LABEL = {
    0: "human",
    1: "ai",
}


##########################################################################
# Dataset Loader
##########################################################################

def _load_dataset(path: Path) -> pd.DataFrame:

    frame = pd.read_csv(path)

    if {"text", "label"} <= set(frame.columns):
        return frame[["text", "label"]].dropna()

    if {"text", "generated"} <= set(frame.columns):
        frame = frame.rename(
            columns={
                "generated": "label"
            }
        )

        frame["label"] = frame["label"].map(
            {
                0: "human",
                1: "ai"
            }
        )

        return frame[["text", "label"]].dropna()

    if {"abstract", "label"} <= set(frame.columns):

        frame = frame.rename(
            columns={
                "abstract": "text"
            }
        )

        frame["label"] = frame["label"].map(
            {
                0: "human",
                1: "ai"
            }
        )

        return frame[["text", "label"]].dropna()

    raise SystemExit(
        f"{path} has unsupported columns."
    )


##########################################################################
# Metrics
##########################################################################

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=1,
    )

    return {
        "accuracy": accuracy_score(
            labels,
            predictions,
        ),
        "f1": f1_score(
            labels,
            predictions,
        ),
    }


##########################################################################
# Main
##########################################################################

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--base-model",
        default="roberta-base",
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "models/artifacts/fine_tuned_ai_detector"
        ),
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--chunk-id",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    ######################################################
    # Load dataset
    ######################################################

    frame = _load_dataset(
        args.data
    )

    frame = frame.sample(
        frac=1,
        random_state=args.seed,
    ).reset_index(
        drop=True
    )

    start = args.chunk_id * args.chunk_size

    end = min(
        start + args.chunk_size,
        len(frame),
    )

    frame = frame.iloc[
        start:end
    ].copy()

    print("=" * 60)
    print(f"Chunk : {args.chunk_id}")
    print(f"Rows  : {start} -> {end-1}")
    print(f"Total : {len(frame)}")
    print("=" * 60)

    frame["label_id"] = frame["label"].map(
        LABEL2ID
    )

    train_frame, eval_frame = train_test_split(
        frame,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=frame["label"],
    )

    ######################################################
    # Tokenizer
    ######################################################

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model
    )

    def tokenize(batch):

        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=512,
        )

    train_dataset = Dataset.from_pandas(
        train_frame[
            ["text", "label_id"]
        ].rename(
            columns={
                "label_id": "labels"
            }
        )
    ).map(
        tokenize,
        batched=True,
    )

    eval_dataset = Dataset.from_pandas(
        eval_frame[
            ["text", "label_id"]
        ].rename(
            columns={
                "label_id": "labels"
            }
        )
    ).map(
        tokenize,
        batched=True,
    )

    data_collator = DataCollatorWithPadding(
        tokenizer
    )
    ######################################################
    # Device
    ######################################################

    if torch.backends.mps.is_available():
        print("\n✅ Apple Silicon GPU (MPS) detected")
        device = torch.device("mps")
    else:
        print("\n⚠️ MPS unavailable, using CPU")
        device = torch.device("cpu")

    ######################################################
    # Model
    ######################################################

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    model.to(device)

    ######################################################
    # Training Arguments
    ######################################################

    checkpoint_dir = args.out / "checkpoints"

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),

        num_train_epochs=args.epochs,

        per_device_train_batch_size=args.batch_size,

        per_device_eval_batch_size=args.batch_size,

        learning_rate=2e-5,

        weight_decay=0.01,

        warmup_ratio=0.10,

        logging_strategy="steps",

        logging_steps=25,

        eval_strategy="epoch",

        save_strategy="epoch",

        save_total_limit=3,

        load_best_model_at_end=True,

        metric_for_best_model="f1",

        greater_is_better=True,

        seed=args.seed,

        report_to="none",
    )

    ######################################################
    # Trainer
    ######################################################

    trainer = Trainer(
        model=model,

        args=training_args,

        train_dataset=train_dataset,

        eval_dataset=eval_dataset,

        tokenizer=tokenizer,

        data_collator=data_collator,

        compute_metrics=compute_metrics,
    )

    ######################################################
    # Resume from Checkpoint
    ######################################################

    last_checkpoint = None

    if checkpoint_dir.exists():

        checkpoints = sorted(
            checkpoint_dir.glob("checkpoint-*"),
            key=lambda x: int(x.name.split("-")[-1]),
        )

        if checkpoints:
            last_checkpoint = str(checkpoints[-1])

            print(
                f"\n📂 Resuming from checkpoint:\n{last_checkpoint}\n"
            )

    ######################################################
    # Train
    ######################################################

    trainer.train(
        resume_from_checkpoint=last_checkpoint
    )

    ######################################################
    # Evaluation
    ######################################################

    print("\nRunning evaluation...\n")

    metrics = trainer.evaluate()

    print(metrics)

    ######################################################
    # Save Final Model
    ######################################################

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    trainer.save_model(
        str(args.out)
    )

    tokenizer.save_pretrained(
        str(args.out)
    )

    print("\n" + "=" * 60)

    print("✅ Fine-tuned model saved successfully")

    print(f"Location : {args.out}")

    print("=" * 60)

    print(
        "\nUpdate your config:\n"
    )

    print(
        "TRANSFORMER_AI_DETECTOR_MODEL="
        + str(args.out)
    )

    print(
        "\nTraining Complete!"
    )


###############################################################
# Entry Point
###############################################################

if __name__ == "__main__":
    main()