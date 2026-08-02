"""FAISS-backed approximate nearest-neighbor vector store.

Replaces the dense NumPy/sklearn cosine-similarity matrix the project used
before. Why: computing cosine_similarity(query, corpus_matrix) is an O(n)
dense matrix multiply per query, holding the whole matrix in memory -- fine
for a few hundred sentences, not for "search across thousands of documents."

IndexHNSWFlat gives approximate nearest-neighbor search via a navigable
small-world graph (sub-linear query time), and -- just as important for
*incremental* indexing -- supports adding vectors after the fact with no
retraining step. That's a real advantage over this project's TF-IDF+LSA
embedder, whose TruncatedSVD must be refit on the whole corpus vocabulary
whenever new documents are added; swapping in a fixed-space embedder (e.g.
Sentence-Transformers, see embeddings/sentence_transformer.py) is what makes
genuinely incremental indexing possible end to end.

Trade-off being made explicitly: HNSW is *approximate* -- it can miss a true
nearest neighbor in exchange for speed. For a corpus in the thousands, exact
search (IndexFlatIP) would still be fast enough and simpler; HNSW is chosen
here because the requirement is to scale toward much larger corpora, where
exact search stops being viable. `ef_search` controls that speed/recall
trade-off directly (higher = slower, closer to exact).

Vectors are expected to already be L2-normalised (every embedder in
embeddings/ normalises its output), so inner-product search is equivalent to
cosine similarity here.
"""
from __future__ import annotations
import pickle
from pathlib import Path

import faiss
import numpy as np


class FaissVectorStore:
    def __init__(self, dim: int, m: int = 32, ef_construction: int = 100, ef_search: int = 64):
        self.dim = dim
        base_index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
        base_index.hnsw.efConstruction = ef_construction
        base_index.hnsw.efSearch = ef_search
        # IDMap2 lets us assign stable integer ids ourselves, so a caller can
        # keep parallel metadata (source document, sentence text) indexed by
        # the same id even as more vectors are added later.
        self._index = faiss.IndexIDMap2(base_index)
        self._next_id = 0

    @property
    def size(self) -> int:
        return self._index.ntotal

    def add(self, vectors: np.ndarray) -> list[int]:
        """Add vectors, returning the ids assigned to them. Ids are never
        reused across calls, so this is safe to call repeatedly as new
        documents arrive (incremental indexing) rather than only once at
        build time.
        """
        if vectors.shape[0] == 0:
            return []
        ids = np.arange(self._next_id, self._next_id + vectors.shape[0]).astype(np.int64)
        self._index.add_with_ids(np.ascontiguousarray(vectors, dtype=np.float32), ids)
        self._next_id += vectors.shape[0]
        return ids.tolist()

    def search(self, query_vectors: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns (scores, ids), each shape (n_queries, top_k). Slots beyond
        how many vectors actually exist are padded with score 0 / id -1
        rather than raising, so callers don't need a special case for a
        small or empty store.
        """
        n_queries = query_vectors.shape[0]
        if self._index.ntotal == 0:
            return np.zeros((n_queries, top_k), dtype=np.float32), -np.ones((n_queries, top_k), dtype=np.int64)

        k = min(top_k, self._index.ntotal)
        scores, ids = self._index.search(np.ascontiguousarray(query_vectors, dtype=np.float32), k)
        if k < top_k:
            pad_scores = np.zeros((n_queries, top_k - k), dtype=np.float32)
            pad_ids = -np.ones((n_queries, top_k - k), dtype=np.int64)
            scores = np.hstack([scores, pad_scores])
            ids = np.hstack([ids, pad_ids])
        return scores, ids

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path))
        with path.with_suffix(path.suffix + ".meta").open("wb") as handle:
            pickle.dump({"dim": self.dim, "next_id": self._next_id}, handle)

    @classmethod
    def load(cls, path: Path) -> "FaissVectorStore":
        path = Path(path)
        store = cls.__new__(cls)
        store._index = faiss.read_index(str(path))
        with path.with_suffix(path.suffix + ".meta").open("rb") as handle:
            meta = pickle.load(handle)
        store.dim = meta["dim"]
        store._next_id = meta["next_id"]
        return store
