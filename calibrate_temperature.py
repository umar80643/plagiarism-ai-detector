"""Fit temperature scaling for the transformer AI-detector on a held-out
labeled set, and save it so TransformerDetector picks it up automatically.

Needs the transformer's actual weights downloaded (see
models/ai_detectors/transformer_backend.py's verification note) and runs
inference over every row of --data, so this is meaningfully slower than
train_ai_detector.py's TF-IDF+LogisticRegression path. Not runnable in the
environment this project was built in (huggingface.co blocked); written and
reviewed for correctness, and models/ai_detectors/calibration.py's actual
math is unit-tested independent of the model (tests/test_calibration.py).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config
from models.ai_detectors.calibration import fit_temperature, negative_log_likelihood
from models.ai_detectors.transformer_backend import TransformerDetector


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
    parser.add_argument("--model-name", default=config.TRANSFORMER_AI_DETECTOR_MODEL)
    parser.add_argument("--out", type=Path, default=config.TRANSFORMER_TEMPERATURE_PATH)
    args = parser.parse_args()

    frame = _load_dataset(args.data)
    detector = TransformerDetector(model_name=args.model_name)
    detector._ensure_loaded()  # surfaces a download/availability error early, with a clear traceback

    logits_list = []
    true_indices = []
    for text, label in zip(frame["text"], frame["label"]):
        inputs = detector._tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
        with detector._torch.no_grad():
            logits = detector._model(**inputs).logits[0].numpy()
        logits_list.append(logits)
        true_indices.append(detector._label_to_index[label])

    logits_array = np.stack(logits_list)
    true_indices_array = np.array(true_indices)

    nll_before = negative_log_likelihood(1.0, logits_array, true_indices_array)
    temperature = fit_temperature(logits_array, true_indices_array)
    nll_after = negative_log_likelihood(temperature, logits_array, true_indices_array)

    print(f"temperature: {temperature:.4f}")
    print(f"NLL before calibration: {nll_before:.4f}")
    print(f"NLL after calibration:  {nll_after:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({
            "temperature": temperature, "model_name": args.model_name, "dataset": str(args.data),
            "nll_before": nll_before, "nll_after": nll_after, "n_examples": len(frame),
        }, indent=2),
        encoding="utf-8",
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
