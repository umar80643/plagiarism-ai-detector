"""Builds the embedder/reranker selected by config.py, so the rest of the
code depends on the TextEmbedder / Reranker interfaces, not on which config
flag or environment variable chose a particular implementation.
"""
from __future__ import annotations

import config
from embeddings.base import TextEmbedder
from embeddings.tfidf_lsa import TfidfLsaEmbedder
from models.reranker import CrossEncoderReranker, LexicalOverlapReranker, Reranker


def build_embedder() -> TextEmbedder:
    if config.EMBEDDING_BACKEND == "sentence_transformer":
        from embeddings.sentence_transformer import MODEL_PRESETS, SentenceTransformerEmbedder
        model_name = MODEL_PRESETS[config.SENTENCE_TRANSFORMER_MODEL_PRESET]
        return SentenceTransformerEmbedder(model_name=model_name)
    return TfidfLsaEmbedder()


def build_reranker() -> Reranker | None:
    if not config.RERANK_ENABLED:
        return None
    if config.RERANKER_BACKEND == "lexical":
        return LexicalOverlapReranker()
    return CrossEncoderReranker()
