"""Text cleaning and sentence splitting.

Sentence splitting uses NLTK's Punkt tokenizer when its data is available
(download once with `python -m nltk.downloader punkt`), and falls back to a
regex splitter otherwise, so the app never hard-crashes on a fresh machine.
"""
from __future__ import annotations
import re

import nltk

_WHITESPACE_RE = re.compile(r"\s+")
_NON_WORD_RE = re.compile(r"[^\w\s]")
_FALLBACK_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace."""
    text = _NON_WORD_RE.sub(" ", text.lower())
    return _WHITESPACE_RE.sub(" ", text).strip()


def split_sentences(text: str) -> list[str]:
    """Split into sentences, preferring NLTK's Punkt tokenizer."""
    text = text.strip()
    if not text:
        return []
    try:
        sentences = nltk.sent_tokenize(text)
    except LookupError:
        sentences = _FALLBACK_SENTENCE_RE.split(text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]
