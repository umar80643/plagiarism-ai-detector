"""Builds the AI-detector backend selected by config.py.

Falls back toward a simpler backend if the configured one isn't actually
usable yet (e.g. AI_DETECTOR_BACKEND=sklearn but train_ai_detector.py hasn't
been run), rather than hard-failing every request -- consistent with how the
rest of this project degrades (see models/ai_text_detector.py's original
heuristic-first design). The fallback is logged, not silent.
"""
from __future__ import annotations
import json
import logging

import config
from models.ai_detectors.base import AIDetectorBackend
from models.ai_detectors.heuristic import HeuristicDetector
from models.ai_detectors.sklearn_backend import SklearnDetector

LOGGER = logging.getLogger(__name__)


def _read_calibrated_temperature() -> float:
    if not config.TRANSFORMER_TEMPERATURE_PATH.exists():
        return 1.0
    data = json.loads(config.TRANSFORMER_TEMPERATURE_PATH.read_text(encoding="utf-8"))
    return float(data["temperature"])


def build_ai_detector() -> AIDetectorBackend:
    backend_name = config.AI_DETECTOR_BACKEND

    if backend_name == "auto":
        sklearn_detector = SklearnDetector()
        return sklearn_detector if sklearn_detector.is_available() else HeuristicDetector()

    if backend_name == "transformer":
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            LOGGER.warning("transformers/torch not installed (see requirements-semantic.txt); falling back to sklearn/heuristic")
            backend_name = "sklearn"
        else:
            from models.ai_detectors.transformer_backend import TransformerDetector
            return TransformerDetector(
                model_name=config.TRANSFORMER_AI_DETECTOR_MODEL,
                temperature=_read_calibrated_temperature(),
            )

    if backend_name == "sklearn":
        sklearn_detector = SklearnDetector()
        if sklearn_detector.is_available():
            return sklearn_detector
        LOGGER.warning("no trained model at %s; falling back to heuristic", sklearn_detector.model_path)

    return HeuristicDetector()
