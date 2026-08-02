"""Loads the reference corpus from disk. Split into its own module so both
plagiarism_model.py and plagiarism_index.py can depend on it without a
circular import.
"""
from __future__ import annotations
from pathlib import Path

from config import CORPUS_DIR
from utils.file_reader import extract_text

SUPPORTED_SUFFIXES = {".txt", ".pdf", ".docx"}


def load_corpus(corpus_dir: Path = CORPUS_DIR) -> dict[str, str]:
    """Read every supported file in corpus_dir into {filename: text}."""
    documents: dict[str, str] = {}
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.exists():
        return documents
    for path in sorted(corpus_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            try:
                documents[path.name] = extract_text(path)
            except Exception:
                continue  # skip unreadable files rather than crash the whole comparison
    return documents
