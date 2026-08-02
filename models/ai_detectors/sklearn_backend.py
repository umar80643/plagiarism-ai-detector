"""TF-IDF + Logistic Regression backend (see train_ai_detector.py). A real
trained classifier, evaluated with a genuine held-out split -- see
models/artifacts/ai_detector_metrics.json after training -- but not a
transformer; kept as a lighter-weight option than TransformerDetector.
"""
from __future__ import annotations
from functools import lru_cache
import pickle
from pathlib import Path

from config import ROOT
from models.ai_detectors.base import AIDetectionResult
from utils.explanation import explain_ai_signals

MODEL_PATH = ROOT / "models" / "artifacts" / "ai_detector.pkl"


@lru_cache(maxsize=1)
def _load_artifact(model_path: Path) -> dict | None:
    if not model_path.exists():
        return None
    with model_path.open("rb") as handle:
        return pickle.load(handle)


def clear_model_cache() -> None:
    """Call after retraining (or in tests) so a newly saved artifact is
    picked up instead of a stale cached one.
    """
    _load_artifact.cache_clear()


class SklearnDetector:
    def __init__(self, model_path: Path | None = None):
        # Deliberately not `model_path: Path = MODEL_PATH` -- a default
        # argument value is computed once, when this class is defined, so
        # monkeypatching the module-level MODEL_PATH afterward (tests, or a
        # config reload) wouldn't affect it. Reading MODEL_PATH here, inside
        # the function body, looks it up fresh from the module's current
        # globals every time a SklearnDetector is constructed.
        self.model_path = model_path if model_path is not None else MODEL_PATH

    def is_available(self) -> bool:
        return _load_artifact(self.model_path) is not None

    def predict(self, text: str) -> AIDetectionResult:
        artifact = _load_artifact(self.model_path)
        if artifact is None:
            raise RuntimeError(f"No trained model at {self.model_path}; run train_ai_detector.py first")

        features = artifact["vectorizer"].transform([text])
        probabilities = artifact["estimator"].predict_proba(features)[0]
        classes = list(artifact["estimator"].classes_)
        ai_probability = float(probabilities[classes.index("ai")])
        human_probability = float(probabilities[classes.index("human")])

        label = "AI" if ai_probability >= 0.5 else "Human"
        confidence = abs(ai_probability - 0.5) * 2
        return AIDetectionResult(
            human_probability=round(human_probability, 4),
            ai_probability=round(ai_probability, 4),
            confidence=round(confidence, 4),
            label=label,
            explanation=explain_ai_signals(text),
            backend="sklearn",
        )
