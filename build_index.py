"""Build (or incrementally update) the cached plagiarism embedding index.

Run this whenever the reference corpus changes. The API and app load the
cached index at startup rather than re-embedding the corpus on every
request.

--incremental only embeds documents that aren't already in the saved index
(matched by filename), rather than rebuilding from scratch -- meaningful
with a fixed-space embedder (Sentence-Transformers); with the default
TF-IDF+LSA embedder it still works, but new documents are embedded using the
vocabulary the index was last fully built with (see PlagiarismIndex.add_documents'
docstring), so periodically doing a full rebuild is still worth doing if
you're not on a fixed-space embedder.
"""
from __future__ import annotations
import argparse
import logging
from pathlib import Path

from config import CORPUS_DIR
from models.corpus import load_corpus
from models.plagiarism_index import INDEX_PATH, PlagiarismIndex, build_index_from_dir

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_DIR)
    parser.add_argument("--out", type=Path, default=INDEX_PATH)
    parser.add_argument("--incremental", action="store_true", help="Only add documents not already in the saved index.")
    args = parser.parse_args()

    if args.incremental and args.out.exists():
        index = PlagiarismIndex.load(args.out)
        full_corpus = load_corpus(args.corpus_dir)
        new_documents = {name: text for name, text in full_corpus.items() if name not in index.document_names}
        if not new_documents:
            LOGGER.info("no new documents in %s; index already up to date", args.corpus_dir)
            return
        index.add_documents(new_documents)
        LOGGER.info("incrementally added %d document(s): %s", len(new_documents), ", ".join(new_documents))
    else:
        index = build_index_from_dir(args.corpus_dir)
        LOGGER.info("built a fresh index from %s", args.corpus_dir)

    index.save(args.out)
    LOGGER.info(
        "indexed %d documents, %d sentences -> %s",
        len(index.document_names), len(index.sentence_texts), args.out,
    )


if __name__ == "__main__":
    main()
