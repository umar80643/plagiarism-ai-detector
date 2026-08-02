"""Shared interface every AI-text-detector backend implements.

The point of this interface: models/ai_text_detector.py, the API, and the
UI depend on AIDetectorBackend, never on "is this the heuristic, the sklearn
model, or the transformer." Swapping which backend is active (config.py) or
adding a new one never requires touching those call sites.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class AIDetectionResult:
    human_probability: float  # 0-1
    ai_probability: float  # 0-1
    confidence: float  # 0-1, how far the winning probability is from a 50/50 toss-up
    label: str  # "AI" or "Human"
    explanation: list[str]  # human-readable reasons, e.g. "low lexical diversity"
    backend: str  # which implementation produced this, e.g. "heuristic", "transformer"


class AIDetectorBackend(Protocol):
    def predict(self, text: str) -> AIDetectionResult: ...
