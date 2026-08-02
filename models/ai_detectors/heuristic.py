"""Heuristic AI-text scorer: lexical diversity + sentence-length uniformity.
No training data or model download needed -- the default when nothing else
is configured/available.
"""
from __future__ import annotations

from config import DEFAULT_AI_DETECTOR_SETTINGS
from models.ai_detectors.base import AIDetectionResult
from utils.explanation import explain_ai_signals
from utils.text_analysis import analyze_text


def _inverse_ratio(value: float, low_threshold: float) -> float:
    """1.0 when value is 0 (maximally 'flat'/uniform), 0.0 at or above the
    threshold (varied enough to look human-like on this signal alone).
    """
    if low_threshold <= 0 or value >= low_threshold:
        return 0.0
    return (low_threshold - value) / low_threshold


class HeuristicDetector:
    def __init__(self, settings=DEFAULT_AI_DETECTOR_SETTINGS):
        self.settings = settings

    def predict(self, text: str) -> AIDetectionResult:
        stats = analyze_text(text)
        diversity_component = _inverse_ratio(stats["lexical_diversity"], self.settings.low_lexical_diversity)
        burstiness_component = _inverse_ratio(stats["burstiness"], self.settings.low_burstiness)
        ai_probability = (diversity_component + burstiness_component) / 2

        label = "AI" if ai_probability >= 0.5 else "Human"
        confidence = abs(ai_probability - 0.5) * 2  # 0 at the 50/50 boundary, 1 at either extreme
        return AIDetectionResult(
            human_probability=round(1 - ai_probability, 4),
            ai_probability=round(ai_probability, 4),
            confidence=round(confidence, 4),
            label=label,
            explanation=explain_ai_signals(text, self.settings),
            backend="heuristic",
        )
