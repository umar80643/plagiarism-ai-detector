"""Descriptive text statistics shared by the UI and the AI-text detector."""
from __future__ import annotations
import statistics

from utils.preprocess import split_sentences

WORDS_PER_MINUTE = 200


def analyze_text(text: str) -> dict:
    sentences = split_sentences(text)
    words = text.split()
    word_count = len(words)
    sentence_count = len(sentences) or 1
    sentence_lengths = [len(sentence.split()) for sentence in sentences] or [word_count]

    unique_words = {word.lower().strip(".,!?;:\"'()") for word in words}
    lexical_diversity = round(len(unique_words) / word_count, 3) if word_count else 0.0

    return {
        "word_count": word_count,
        "sentence_count": len(sentences),
        "reading_time": max(1, round(word_count / WORDS_PER_MINUTE)),
        "avg_sentence_length": round(word_count / sentence_count, 2),
        "lexical_diversity": lexical_diversity,
        # Standard deviation of sentence length in words. Human writing tends
        # to vary sentence length more ("burstiness"); very uniform sentence
        # lengths are one heuristic signal of machine-generated text.
        "burstiness": round(statistics.pstdev(sentence_lengths), 2) if len(sentence_lengths) > 1 else 0.0,
    }
