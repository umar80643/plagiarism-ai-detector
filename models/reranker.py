"""Two-stage retrieve-then-rerank: FAISS does cheap approximate retrieval
over the whole corpus, then a cross-encoder does expensive, precise scoring
of just the top-K candidates. This is the standard architecture real search
systems use (a bi-encoder alone trades accuracy for speed; a cross-encoder
alone doesn't scale past a tiny candidate set) -- doing both, in that order,
gets both properties.

Why a cross-encoder is more accurate than the bi-encoder (embedding cosine
similarity) already used for retrieval: a bi-encoder embeds the query and
each candidate *independently*, so it never lets one influence how the other
is represented. A cross-encoder feeds (query, candidate) through the model
*together*, so attention can directly relate specific words across the pair
-- more accurate, but it must run once per candidate, which is why it only
reranks the FAISS-shortlisted top-K rather than searching the whole corpus.
"""
from __future__ import annotations
from typing import Protocol, Sequence

import numpy as np


class Reranker(Protocol):
    def rerank(self, query: str, candidates: Sequence[str]) -> list[tuple[int, float]]:
        """Returns (original_index_into_candidates, score) pairs, sorted by
        score descending.
        """
        ...


class CrossEncoderReranker:
    """Wraps a HuggingFace cross-encoder (default: cross-encoder/ms-marco-MiniLM-L-6-v2,
    trained for exactly this query-vs-passage relevance task).

    NOTE ON VERIFICATION: same caveat as SentenceTransformerEmbedder -- this
    downloads weights from huggingface.co on first use, which this project
    was built without access to; verify with `pytest -m live_model` on a
    machine with normal internet access.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, candidates: Sequence[str]) -> list[tuple[int, float]]:
        if not candidates:
            return []
        model = self._ensure_loaded()
        raw_scores = np.asarray(model.predict([[query, candidate] for candidate in candidates]))
        # ms-marco cross-encoders output an unbounded relevance logit, not a
        # 0-1 similarity; squash it so callers can treat every Reranker's
        # output as roughly comparable, percentage-like magnitude. This is a
        # deliberate semantic shift worth stating plainly: after reranking,
        # the resulting "similarity" represents cross-encoder relevance
        # confidence, not the cosine similarity FAISS retrieval used -- a
        # different (more accurate for this purpose) notion of "match".
        scores = 1.0 / (1.0 + np.exp(-raw_scores))
        order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
        return [(i, float(scores[i])) for i in order]


class LexicalOverlapReranker:
    """A dependency-free fallback: scores candidates by word (Jaccard)
    overlap with the query instead of a learned cross-encoder.

    This exists for two reasons: (1) it keeps the retrieve-then-rerank
    *pipeline* -- not just the retrieval step -- exercised by tests without
    a model download, and (2) it's a legitimate (if meaningfully weaker)
    fallback for fully offline deployments where downloading cross-encoder
    weights isn't an option at all. It is not a substitute for
    CrossEncoderReranker's accuracy; it exists so the pipeline degrades
    gracefully rather than being unavailable.
    """

    def rerank(self, query: str, candidates: Sequence[str]) -> list[tuple[int, float]]:
        if not candidates:
            return []
        query_words = set(query.lower().split())
        scores = []
        for candidate in candidates:
            candidate_words = set(candidate.lower().split())
            union = query_words | candidate_words
            scores.append(len(query_words & candidate_words) / len(union) if union else 0.0)
        order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
        return [(i, scores[i]) for i in order]
