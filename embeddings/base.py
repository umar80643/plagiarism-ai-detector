"""Shared interface for text embedders, so plagiarism detection can swap
backends (TF-IDF+LSA today, sentence-transformers later) without touching
the code that uses embeddings.
"""
from __future__ import annotations
from typing import Protocol, Sequence

import numpy as np


class TextEmbedder(Protocol):
    """fit() on a reference corpus once; transform() any new text into that
    same vector space afterwards. Cosine similarity between transform()
    outputs is what plagiarism detection and AI-detector features use.
    """

    def fit(self, texts: Sequence[str]) -> "TextEmbedder": ...

    def transform(self, texts: Sequence[str]) -> np.ndarray: ...
