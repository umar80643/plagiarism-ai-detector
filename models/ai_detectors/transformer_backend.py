"""Transformer-based AI-text detector via HuggingFace Transformers.

Model choice, explained: roberta-base and microsoft/deberta-v3-base, used
directly for sequence classification, load with a *randomly initialized*
classification head -- they're pretrained language models, not fine-tuned
AI-text detectors. Wiring one up untrained and calling it "AI detection"
would score at chance, worse than the existing heuristic. The honest default
here is therefore "openai-community/roberta-base-openai-detector" -- OpenAI's
own real, publicly released, fine-tuned checkpoint (trained to detect GPT-2
output specifically).

Trade-off, stated plainly: that checkpoint is several years old and trained
against GPT-2, so it's meaningfully weaker against modern LLM output than a
model fine-tuned on current generations would be. `train_transformer_ai_detector.py`
fine-tunes roberta-base or microsoft/deberta-v3-base on your own labeled data
for exactly that reason -- "support future fine-tuning" per the requirement.
Swap MODEL_NAME (config.TRANSFORMER_AI_DETECTOR_MODEL) to a fine-tuned
checkpoint's path/name and nothing else in this file, the API, or the UI
needs to change.

Calibration: raw softmax probabilities from a fine-tuned transformer are
frequently overconfident (Guo et al., 2017, "On Calibration of Modern Neural
Networks"). Temperature scaling divides the pre-softmax logits by a single
learned scalar T > 1 before the softmax, which smooths the distribution
without changing which class wins -- see calibrate_temperature.py to fit T
on a held-out labeled set; T=1.0 (no-op) is the default until that's run.

NOTE ON VERIFICATION: same caveat as the Sentence-Transformers/cross-encoder
classes in embeddings/ and models/reranker.py -- this downloads weights from
huggingface.co on first use, unreachable in the environment this was built
in. Written and reviewed carefully; verify with `pytest -m live_model` on a
machine with normal internet access.
"""
from __future__ import annotations

import numpy as np

from models.ai_detectors.base import AIDetectionResult
from utils.explanation import explain_ai_signals

DEFAULT_MODEL_NAME = "openai-community/roberta-base-openai-detector"


class TransformerDetector:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, temperature: float = 1.0):
        self.model_name = model_name
        self.temperature = temperature
        self._model = None
        self._tokenizer = None
        self._label_to_index: dict[str, int] | None = None

    def _ensure_loaded(self):
        if self._model is None:
            # Heavy, optional dependency: imported lazily so importing this
            # module (or the package it's in) never triggers a download.
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._torch = torch
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._model.eval()
            # Different checkpoints label their classes differently (e.g.
            # "Real"/"Fake", "human"/"ai", "LABEL_0"/"LABEL_1") -- read the
            # model's own id2label mapping rather than assuming an order, so
            # swapping in a different fine-tuned checkpoint doesn't silently
            # flip predictions.
            id2label = {k: v.lower() for k, v in self._model.config.id2label.items()}
            self._label_to_index = {}
            for index, label in id2label.items():
                if any(token in label for token in ("fake", "ai", "machine", "generated")):
                    self._label_to_index["ai"] = index
                elif any(token in label for token in ("real", "human")):
                    self._label_to_index["human"] = index
            if set(self._label_to_index) != {"ai", "human"}:
                raise ValueError(
                    f"Could not map {self.model_name}'s labels {id2label} to ai/human; "
                    "set the mapping explicitly for this checkpoint."
                )
        return self._model

    def predict(self, text: str) -> AIDetectionResult:
        model = self._ensure_loaded()
        torch = self._torch
        inputs = self._tokenizer(text, truncation=True, max_length=512, return_tensors="pt")

        with torch.no_grad():
            logits = model(**inputs).logits[0].numpy()

        calibrated_logits = logits / self.temperature
        probabilities = _softmax(calibrated_logits)

        ai_probability = float(probabilities[self._label_to_index["ai"]])
        human_probability = float(probabilities[self._label_to_index["human"]])
        label = "AI" if ai_probability >= 0.5 else "Human"
        confidence = abs(ai_probability - 0.5) * 2

        explanation = explain_ai_signals(text)
        explanation.append(
            f"transformer classifier ({self.model_name.split('/')[-1]}) predicts "
            f"{'AI-generated' if label == 'AI' else 'human-written'} with {max(ai_probability, human_probability) * 100:.1f}% probability"
        )
        return AIDetectionResult(
            human_probability=round(human_probability, 4),
            ai_probability=round(ai_probability, 4),
            confidence=round(confidence, 4),
            label=label,
            explanation=explanation,
            backend="transformer",
        )


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)  # numerically stable softmax
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum()
