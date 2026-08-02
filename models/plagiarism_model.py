"""Corpus-based plagiarism detection using a precomputed FAISS embedding index.

Compares an input document against a *reference corpus* -- the part the
original version of this project was missing entirely (it only compared a
document's sentences to itself, which can't detect copying from an external
source at all). The embedding backend is configurable (config.EMBEDDING_BACKEND):
TF-IDF+LSA by default (fully offline), or Sentence-Transformers for stronger
paraphrase detection (see embeddings/sentence_transformer.py) -- either way,
similarity is computed in an embedding space, not raw word overlap, so
paraphrased content scores higher than exact-word TF-IDF alone would give it.
"""
from __future__ import annotations
from pathlib import Path

from config import CORPUS_DIR
from models.corpus import load_corpus
from models.plagiarism_index import INDEX_PATH, PlagiarismIndex, build_index, build_index_from_dir

__all__ = ["load_corpus", "detect_plagiarism"]


def _get_index(corpus_dir: Path, extra_documents: dict[str, str] | None) -> PlagiarismIndex:
    if extra_documents:
        # Ad hoc documents mean we can't use the cached, corpus-only index --
        # build a temporary one covering corpus_dir + the extra documents.
        corpus = load_corpus(corpus_dir)
        corpus.update(extra_documents)
        return build_index(corpus)
    if INDEX_PATH.exists():
        return PlagiarismIndex.load(INDEX_PATH)
    # No prebuilt index yet -- build on the fly so the app still works, but
    # this is the slow path; run `python build_index.py` to cache it.
    return build_index_from_dir(corpus_dir)


def detect_plagiarism(
    text: str, corpus_dir: Path = CORPUS_DIR, extra_documents: dict[str, str] | None = None
) -> dict:
    """Returns:
        {
          "score": float,             # highest whole-document similarity, 0-100
          "matches": [{"source", "similarity"}, ...],  # every corpus doc, sorted
          "sentence_matches": [{"sentence", "source", "matched_sentence", "similarity"}, ...],
        }
    """
    index = _get_index(corpus_dir, extra_documents)
    matches = index.score_document(text)
    score = matches[0]["similarity"] if matches else 0.0
    sentence_matches = index.score_sentences(text)
    return {"score": score, "matches": matches, "sentence_matches": sentence_matches}
