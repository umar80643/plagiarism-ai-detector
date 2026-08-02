"""TF-IDF + Latent Semantic Analysis (truncated SVD) embedder.

Raw TF-IDF cosine similarity only catches near-verbatim overlap: two
sentences with the same *meaning* but different wording score low, because
TF-IDF vectors only share dimensions for words that literally match. LSA
projects TF-IDF vectors into a lower-dimensional space built from the
corpus's co-occurrence structure, so synonymous/paraphrased text ends up
closer together -- a real (if older-generation) embedding technique, and one
that needs no model download or GPU, unlike sentence-transformers.

This is the default embedder specifically because it works fully offline.
See embeddings/base.py -- swap in a sentence-transformers-backed embedder
(same fit/transform interface) for stronger paraphrase detection if network
access to download model weights is available in your deployment.
"""
from __future__ import annotations
from typing import Sequence

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import normalize


class TfidfLsaEmbedder:
    def __init__(self, n_components: int = 100, random_state: int = 42):
        self.requested_components = n_components
        self.random_state = random_state
        self._pipeline: Pipeline | None = None

    def fit(self, texts: Sequence[str]) -> "TfidfLsaEmbedder":
        texts = list(texts)
        vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
        tfidf = vectorizer.fit_transform(texts)
        # n_components must be strictly less than min(n_samples, n_features);
        # fall back gracefully for tiny corpora (e.g. in unit tests).
        max_components = max(1, min(tfidf.shape) - 1)
        n_components = max(1, min(self.requested_components, max_components))
        svd = TruncatedSVD(n_components=n_components, random_state=self.random_state)
        self._pipeline = Pipeline([("tfidf", vectorizer), ("svd", svd)])
        self._pipeline.fit(texts)
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        if self._pipeline is None:
            raise RuntimeError("TfidfLsaEmbedder.fit() must be called before transform()")
        vectors = self._pipeline.transform(list(texts))
        return normalize(vectors)
