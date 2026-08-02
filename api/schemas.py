"""Request/response schemas for the plagiarism + AI-detection API."""
from __future__ import annotations
from pydantic import BaseModel, Field


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50_000)


class PlagiarismMatch(BaseModel):
    source: str
    similarity: float


class SentenceMatch(BaseModel):
    sentence: str
    source: str | None
    matched_sentence: str | None
    similarity: float


class PlagiarismResponse(BaseModel):
    score: float
    matches: list[PlagiarismMatch]
    sentence_matches: list[SentenceMatch]


class AITextResponse(BaseModel):
    score: float  # unchanged: 0-100, kept for backward compatibility
    label: str  # unchanged: "AI" or "Human"
    human_probability: float
    ai_probability: float
    confidence: float
    explanation: list[str]
    backend: str  # "heuristic", "sklearn", or "transformer" -- which detector produced this


class HealthResponse(BaseModel):
    status: str
    plagiarism_index_documents: int
    ai_detector_mode: str
