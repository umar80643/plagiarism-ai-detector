"""Sentence-Transformers embedder -- a fixed embedding space, unlike TF-IDF+LSA.

Architectural decision worth calling out explicitly: TfidfLsaEmbedder.fit()
must see the whole corpus vocabulary before it can transform anything
(TruncatedSVD is fit on a term-document matrix). Adding a document to the
corpus technically invalidates that embedding space, so "incremental
indexing" on top of it really means "refit everything, cheaply."

A pretrained Sentence-Transformer has no such dependency: it maps any text
into the same fixed-dimension space regardless of what else has been
embedded. fit() here is a deliberate no-op (kept only so this class
satisfies the same TextEmbedder protocol as TfidfLsaEmbedder) -- transform()
works on one new document at any time, directly comparable to everything
already indexed. That's what makes genuinely incremental indexing possible.

Trade-off, stated plainly: this requires downloading pretrained weights
(~420MB for all-mpnet-base-v2, ~130MB for bge-small) from HuggingFace on
first use, and CPU inference is meaningfully slower than TF-IDF+LSA's sparse
matrix ops. MODEL_PRESETS below offers the smaller/faster option for
memory-constrained deployments, per the requirement.

NOTE ON VERIFICATION: this class was written and reviewed carefully, but
could not be executed end-to-end in the environment this was built in --
that environment's network policy blocks huggingface.co, so the model
weights this class downloads on first use were unreachable there. It should
work as-is on any machine with normal internet access; run
`pytest -m live_model` (see tests/test_sentence_transformer_embedder.py) to
verify on yours.
"""
from __future__ import annotations
from typing import Sequence

import numpy as np

MODEL_PRESETS = {
    "quality": "sentence-transformers/all-mpnet-base-v2",  # 768-dim, best accuracy
    "small": "BAAI/bge-small-en-v1.5",  # 384-dim, ~3x smaller/faster, for memory-constrained deployments
}


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = MODEL_PRESETS["quality"], batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None  # lazy: importing this module must never trigger a download

    def _ensure_loaded(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # heavy optional dependency, imported lazily
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def fit(self, texts: Sequence[str]) -> "SentenceTransformerEmbedder":
        """No-op by design (see module docstring). Still loads the model, so
        a caller who wants to surface a download/availability error early
        can do so by calling fit() before transform().
        """
        self._ensure_loaded()
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        model = self._ensure_loaded()
        vectors = model.encode(
            list(texts), batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vectors, dtype=np.float32)
