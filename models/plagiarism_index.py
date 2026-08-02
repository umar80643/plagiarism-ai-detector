"""A precomputed, incrementally-updatable embedding index over the reference
corpus, backed by FAISS (models/vector_store.py) and an optional cross-encoder
reranking stage (models/reranker.py).

Two-stage retrieval: FAISS does cheap approximate search to shortlist the
top-K candidate sentences, then -- if a reranker is configured -- the
cross-encoder re-scores just those K precisely. This is the standard IR
pattern: a bi-encoder (the embedder) alone is fast but less accurate; a
cross-encoder alone is accurate but too slow to run against the whole
corpus. Doing both, in that order, gets both properties.
"""
from __future__ import annotations
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import config
from embeddings.base import TextEmbedder
from models.corpus import load_corpus
from models.reranker import Reranker
from models.vector_store import FaissVectorStore
from utils.preprocess import split_sentences

INDEX_PATH = config.ROOT / "models" / "artifacts" / "plagiarism_index.pkl"


@dataclass
class PlagiarismIndex:
    embedder: TextEmbedder
    document_store: FaissVectorStore
    document_names: list[str]
    sentence_store: FaissVectorStore
    sentence_sources: list[str]
    sentence_texts: list[str]
    reranker: Reranker | None = None
    rerank_top_k: int = 20

    def score_document(self, text: str) -> list[dict]:
        if not self.document_names:
            return []
        vector = self.embedder.transform([text])
        scores, ids = self.document_store.search(vector, top_k=len(self.document_names))
        matches = [
            {"source": self.document_names[idx], "similarity": round(float(score) * 100, 2)}
            for score, idx in zip(scores[0], ids[0])
            if idx != -1
        ]
        return sorted(matches, key=lambda match: match["similarity"], reverse=True)

    def score_sentences(self, text: str) -> list[dict]:
        sentences = split_sentences(text)
        if not sentences:
            return []
        if not self.sentence_texts:
            return [{"sentence": s, "source": None, "matched_sentence": None, "similarity": 0.0} for s in sentences]

        top_k = min(self.rerank_top_k, len(self.sentence_texts)) if self.reranker else 1
        vectors = self.embedder.transform(sentences)
        scores, ids = self.sentence_store.search(vectors, top_k=top_k)

        results = []
        for sentence, score_row, id_row in zip(sentences, scores, ids):
            candidate_ids = [int(i) for i in id_row if i != -1]
            if not candidate_ids:
                results.append({"sentence": sentence, "source": None, "matched_sentence": None, "similarity": 0.0})
                continue

            if self.reranker and len(candidate_ids) > 1:
                candidate_texts = [self.sentence_texts[i] for i in candidate_ids]
                reranked = self.reranker.rerank(sentence, candidate_texts)
                best_local_index, similarity_fraction = reranked[0]
                best_id = candidate_ids[best_local_index]
            else:
                best_id = candidate_ids[0]
                similarity_fraction = float(score_row[0])

            results.append({
                "sentence": sentence,
                "source": self.sentence_sources[best_id],
                "matched_sentence": self.sentence_texts[best_id],
                "similarity": round(similarity_fraction * 100, 2),
            })
        return results

    def add_documents(self, new_corpus: dict[str, str]) -> None:
        """Incrementally index new documents without rebuilding the whole
        index. Safe and exact with a fixed-space embedder (Sentence-
        Transformers); mechanically works with TfidfLsaEmbedder too (it
        transforms new text using the vocabulary already fit), but any words
        not seen at the last full build() are silently dropped from the new
        documents' vectors -- a real accuracy trade-off, not a bug, and the
        reason a fixed-space embedder is what actually delivers on
        "incremental indexing."
        """
        if not new_corpus:
            return
        new_document_names = list(new_corpus.keys())
        document_vectors = self.embedder.transform(list(new_corpus.values()))
        self.document_store.add(document_vectors)
        self.document_names.extend(new_document_names)

        new_sentence_sources: list[str] = []
        new_sentence_texts: list[str] = []
        for name, text in new_corpus.items():
            for sentence in split_sentences(text):
                new_sentence_sources.append(name)
                new_sentence_texts.append(sentence)
        if new_sentence_texts:
            sentence_vectors = self.embedder.transform(new_sentence_texts)
            self.sentence_store.add(sentence_vectors)
            self.sentence_sources.extend(new_sentence_sources)
            self.sentence_texts.extend(new_sentence_texts)

    def save(self, path: Path = INDEX_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.document_store.save(path.with_suffix(".documents.faiss"))
        self.sentence_store.save(path.with_suffix(".sentences.faiss"))
        with path.open("wb") as handle:
            pickle.dump({
                "embedder": self.embedder,
                "document_names": self.document_names,
                "sentence_sources": self.sentence_sources,
                "sentence_texts": self.sentence_texts,
                "reranker": self.reranker,
                "rerank_top_k": self.rerank_top_k,
            }, handle)

    @staticmethod
    def load(path: Path = INDEX_PATH) -> "PlagiarismIndex":
        path = Path(path)
        with path.open("rb") as handle:
            state = pickle.load(handle)
        return PlagiarismIndex(
            embedder=state["embedder"],
            document_store=FaissVectorStore.load(path.with_suffix(".documents.faiss")),
            document_names=state["document_names"],
            sentence_store=FaissVectorStore.load(path.with_suffix(".sentences.faiss")),
            sentence_sources=state["sentence_sources"],
            sentence_texts=state["sentence_texts"],
            reranker=state["reranker"],
            rerank_top_k=state["rerank_top_k"],
        )


def _embedding_dim(embedder: TextEmbedder, sample_texts: list[str]) -> int:
    probe_texts = sample_texts[:1] or ["placeholder text used only to determine embedding dimensionality"]
    return embedder.transform(probe_texts).shape[1]


def build_index(
    corpus: dict[str, str],
    embedder: TextEmbedder | None = None,
    reranker: Reranker | None = None,
    rerank_top_k: int = 20,
) -> PlagiarismIndex:
    """Fit one embedder on the corpus (a no-op for fixed-space embedders,
    see embeddings/sentence_transformer.py), embed every document and every
    sentence, and load both into FAISS stores.
    """
    from embeddings.factory import build_embedder, build_reranker  # local import: avoids a config-time import cycle

    embedder = embedder or build_embedder()
    reranker = reranker if reranker is not None else build_reranker()

    document_names = list(corpus.keys())
    document_texts = list(corpus.values())

    sentence_sources: list[str] = []
    sentence_texts: list[str] = []
    for name, text in corpus.items():
        for sentence in split_sentences(text):
            sentence_sources.append(name)
            sentence_texts.append(sentence)

    fit_corpus = document_texts + sentence_texts
    if fit_corpus:
        embedder.fit(fit_corpus)
        dim = _embedding_dim(embedder, fit_corpus)
    else:
        dim = 1  # arbitrary: an empty corpus never adds vectors to either store
    document_store = FaissVectorStore(dim=dim)
    sentence_store = FaissVectorStore(dim=dim)
    if document_texts:
        document_store.add(embedder.transform(document_texts))
    if sentence_texts:
        sentence_store.add(embedder.transform(sentence_texts))

    return PlagiarismIndex(
        embedder=embedder,
        document_store=document_store,
        document_names=document_names,
        sentence_store=sentence_store,
        sentence_sources=sentence_sources,
        sentence_texts=sentence_texts,
        reranker=reranker,
        rerank_top_k=rerank_top_k,
    )


def build_index_from_dir(corpus_dir: Path = config.CORPUS_DIR, **kwargs) -> PlagiarismIndex:
    return build_index(load_corpus(corpus_dir), **kwargs)
