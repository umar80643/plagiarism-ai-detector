"""Public entry point for AI-text detection.

Delegates to whichever backend config.AI_DETECTOR_BACKEND selects
(models/ai_detectors/factory.py) -- heuristic, the trained sklearn
classifier, or a real transformer. Existing callers (the original
detect_ai_text(text) -> (score, label) signature, used by app.py and the
API) keep working unchanged; detect_ai_text_detailed() is the new,
additive entry point exposing probabilities/confidence/explanation for
anything that wants the richer result.

Runtime fallback: if the configured backend raises (e.g. a transformer
whose weights can't be reached over the network at call time, not just at
startup), this logs the failure and falls back to the heuristic backend
rather than returning an error to the caller -- consistent with how this
project has always preferred to degrade gracefully over hard-failing.
"""
from __future__ import annotations
import logging

from models.ai_detectors.base import AIDetectionResult
from models.ai_detectors.factory import build_ai_detector
from models.ai_detectors.heuristic import HeuristicDetector
from models.ai_detectors.sklearn_backend import MODEL_PATH, clear_model_cache  # re-exported for backward compatibility

__all__ = ["MODEL_PATH", "clear_model_cache", "detect_ai_text", "detect_ai_text_detailed", "AIDetectionResult"]

LOGGER = logging.getLogger(__name__)
_fallback_detector = HeuristicDetector()


def detect_ai_text_detailed(text: str) -> AIDetectionResult:
    detector = build_ai_detector()
    try:
        return detector.predict(text)
    except Exception:
        LOGGER.exception("%s backend failed; falling back to heuristic", getattr(detector, "__class__", type(detector)).__name__)
        return _fallback_detector.predict(text)


def detect_ai_text(text: str) -> tuple[float, str]:
    """Returns (score 0-100, label "AI" or "Human") -- unchanged signature
    for backward compatibility with existing callers.
    """
    result = detect_ai_text_detailed(text)
    return round(result.ai_probability * 100, 2), result.label
