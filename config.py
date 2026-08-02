"""Central paths and thresholds for the plagiarism / AI-text detector."""
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"  # reference documents to check uploads against

EmbeddingBackend = Literal["tfidf_lsa", "sentence_transformer"]

# Env-var driven rather than hard-coded, so a deployment can opt into the
# semantic backend (which needs internet access on first run to download
# model weights) without a code change -- and so this project keeps working
# fully offline by default, since tfidf_lsa needs nothing beyond what's
# already pip-installed.
EMBEDDING_BACKEND: EmbeddingBackend = os.environ.get("EMBEDDING_BACKEND", "tfidf_lsa")  # type: ignore[assignment]
SENTENCE_TRANSFORMER_MODEL_PRESET = os.environ.get("SENTENCE_TRANSFORMER_MODEL_PRESET", "quality")  # "quality" | "small"
RERANK_ENABLED = os.environ.get("RERANK_ENABLED", "false").lower() == "true"
RERANK_TOP_K = int(os.environ.get("RERANK_TOP_K", "20"))
# Which reranker backs RERANK_ENABLED: the real cross-encoder (needs a model
# download) or the dependency-free lexical fallback (works fully offline,
# meaningfully less accurate).
RERANKER_BACKEND: Literal["cross_encoder", "lexical"] = os.environ.get("RERANKER_BACKEND", "cross_encoder")  # type: ignore[assignment]

AIDetectorBackendName = Literal["auto", "heuristic", "sklearn", "transformer"]

# "auto" (the default) preserves this project's original behavior from
# before backends were made pluggable: use the trained sklearn model
# automatically if train_ai_detector.py has been run, otherwise fall back to
# the heuristic -- no configuration needed for that upgrade path. Explicitly
# set "heuristic" or "sklearn" to force one or the other. "transformer" is
# never auto-selected (unlike sklearn) since it requires a large model
# download and heavy optional dependencies -- consistent with
# EMBEDDING_BACKEND defaulting to the fully-offline option rather than
# auto-upgrading to Sentence-Transformers.
AI_DETECTOR_BACKEND: AIDetectorBackendName = os.environ.get("AI_DETECTOR_BACKEND", "auto")  # type: ignore[assignment]
TRANSFORMER_AI_DETECTOR_MODEL = os.environ.get("TRANSFORMER_AI_DETECTOR_MODEL", "openai-community/roberta-base-openai-detector")
TRANSFORMER_TEMPERATURE_PATH = ROOT / "models" / "artifacts" / "ai_detector_temperature.json"


@dataclass(frozen=True)
class PlagiarismSettings:
    # Whole-document embedding cosine similarity against a corpus document at
    # or above this is reported as a "match" for that source.
    match_threshold: float = 0.35
    # A sentence is flagged as plagiarised if its best match against any
    # corpus sentence is at or above this.
    sentence_match_threshold: float = 0.7
    # A sentence is flagged as internally repeated if its best match against
    # another sentence in the *same* document is at or above this.
    self_repetition_threshold: float = 0.75


@dataclass(frozen=True)
class AIDetectorSettings:
    # Below this type-token ratio, text reads as unusually repetitive for its
    # length -- one signal used by real detectors (e.g. GPTZero, GLTR).
    low_lexical_diversity: float = 0.4
    # AI-generated text tends to have low sentence-length variance
    # ("burstiness" is a term from Gehrmann et al.'s GLTR work and GPTZero);
    # human writing mixes short and long sentences more.
    low_burstiness: float = 4.0
    ai_score_high: float = 70.0
    ai_score_very_high: float = 90.0


DEFAULT_PLAGIARISM_SETTINGS = PlagiarismSettings()
DEFAULT_AI_DETECTOR_SETTINGS = AIDetectorSettings()
