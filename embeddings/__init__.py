"""Text embedding backends behind a common fit/transform interface."""
from embeddings.base import TextEmbedder
from embeddings.tfidf_lsa import TfidfLsaEmbedder

__all__ = ["TextEmbedder", "TfidfLsaEmbedder"]