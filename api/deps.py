"""Singletons loaded once (at startup / first use), not per request.

The plagiarism embedding index is the expensive resource here -- rebuilding
it per request would mean re-embedding the whole corpus on every call.
"""
from __future__ import annotations
from functools import lru_cache

from models.ai_detectors.factory import build_ai_detector
from models.plagiarism_index import INDEX_PATH, PlagiarismIndex, build_index_from_dir


@lru_cache(maxsize=1)
def get_plagiarism_index() -> PlagiarismIndex:
    if INDEX_PATH.exists():
        return PlagiarismIndex.load(INDEX_PATH)
    return build_index_from_dir()


def ai_detector_mode() -> str:
    """Which backend build_ai_detector() would actually hand back right now
    -- reflects config.AI_DETECTOR_BACKEND *and* whatever's actually
    available (e.g. "auto" reporting "sklearn" once a model's been trained).
    """
    return build_ai_detector().__class__.__name__.replace("Detector", "").lower()
