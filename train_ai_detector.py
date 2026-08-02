"""Train a real human-vs-AI text classifier with a genuine held-out
evaluation, replacing models/ai_text_detector.py's heuristic scoring the
moment a trained classifier is saved to models/artifacts/ai_detector.pkl.

Accepts either:
- a CSV with "text" and "label" columns (label in {"human", "ai"}), e.g. the
  bundled data/ai_detector_demo.csv toy set, or
- the AI-GA dataset's "title,abstract,label" schema (label 0/1), fetched via
  scripts/fetch_ai_ga_dataset.py -- 28,662 real human/AI-labeled academic
  abstracts. This is a genuine dataset, not a demo, but note the domain:
  it's academic abstracts, so a classifier trained on it will generalise
  best to similarly formal writing, not casual/social text.

Reports accuracy/precision/recall/F1/ROC-AUC and a confusion matrix on a
held-out test split (not just cross-validation), then refits on the full
dataset for the deployed artifact -- standard practice: evaluate honestly on
data the model never saw, then use everything available for the shipped
model.
"""
from __future__ import annotations
import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

from models.ai_text_detector import MODEL_PATH

METRICS_PATH = MODEL_PATH.parent / "ai_detector_metrics.json"


def _load_dataset(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if {"text", "label"} <= set(frame.columns):
        return frame[["text", "label"]].dropna()
    if {"abstract", "label"} <= set(frame.columns):
        frame = frame.rename(columns={"abstract": "text"})
        frame["label"] = frame["label"].map({0: "human", 1: "ai"}).fillna(frame["label"])
        return frame[["text", "label"]].dropna()
    raise SystemExit(f"{path} must have either 'text,label' or 'abstract,label' columns")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data/ai_detector_demo.csv"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = _load_dataset(args.data)
    train_frame, test_frame = train_test_split(
        frame, test_size=args.test_size, stratify=frame["label"], random_state=args.seed
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95)
    train_features = vectorizer.fit_transform(train_frame["text"])
    test_features = vectorizer.transform(test_frame["text"])

    estimator = LogisticRegression(max_iter=1000, class_weight="balanced")
    estimator.fit(train_features, train_frame["label"])

    predicted = estimator.predict(test_features)
    probabilities = estimator.predict_proba(test_features)
    labels = list(estimator.classes_)
    ai_index = labels.index("ai")

    metrics = {
        "dataset": str(args.data),
        "train_size": len(train_frame),
        "test_size": len(test_frame),
        "accuracy": round(accuracy_score(test_frame["label"], predicted), 4),
        "precision": round(precision_score(test_frame["label"], predicted, pos_label="ai"), 4),
        "recall": round(recall_score(test_frame["label"], predicted, pos_label="ai"), 4),
        "f1": round(f1_score(test_frame["label"], predicted, pos_label="ai"), 4),
        "roc_auc": round(
            roc_auc_score((test_frame["label"] == "ai").astype(int), probabilities[:, ai_index]), 4
        ),
        "labels": labels,
        "confusion_matrix": confusion_matrix(test_frame["label"], predicted, labels=labels).tolist(),
    }
    print(json.dumps(metrics, indent=2))

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nsaved held-out evaluation to {METRICS_PATH}")

    # Refit on all available data for the deployed artifact -- the held-out
    # split above is only for honest evaluation, not for the shipped model.
    all_features = vectorizer.fit_transform(frame["text"])
    estimator.fit(all_features, frame["label"])
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as handle:
        pickle.dump({"vectorizer": vectorizer, "estimator": estimator}, handle)
    print(f"saved deployed model (trained on all {len(frame)} rows) to {MODEL_PATH}")


if __name__ == "__main__":
    main()
