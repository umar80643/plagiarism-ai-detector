"""TF-IDF cosine similarity, at the whole-document and sentence level.

Two distinct things live here, and the redesign's main job is to stop
conflating them (the original version's naming made that easy to do):

- calculate_cosine_similarity(): compares TWO documents against each other.
  This is what real plagiarism checking needs -- a document versus some
  other source.
- sentence_similarity_analysis(): finds near-duplicate sentences WITHIN one
  document. That's a legitimate feature (catching self-plagiarism / repeated
  filler), but it is not, by itself, plagiarism detection -- it never looks
  outside the document. models/plagiarism_model.py uses both, clearly
  labelled, rather than presenting the internal check as the whole thing.
"""
from __future__ import annotations
from typing import Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.preprocess import clean_text, split_sentences


def calculate_cosine_similarity(text1: str, text2: str) -> float:
    """Whole-document TF-IDF cosine similarity between two texts, as a
    0-100 percentage.
    """
    cleaned = [clean_text(text1), clean_text(text2)]
    if not cleaned[0] or not cleaned[1]:
        return 0.0
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(cleaned)
    return round(float(cosine_similarity(matrix[0], matrix[1])[0][0]) * 100, 2)


def _similarity_matrix(sentences_a: Sequence[str], sentences_b: Sequence[str]) -> list[list[float]]:
    """Pairwise TF-IDF cosine similarity between two lists of sentences,
    fit on their combined vocabulary so scores are comparable.
    """
    cleaned_a = [clean_text(sentence) for sentence in sentences_a]
    cleaned_b = [clean_text(sentence) for sentence in sentences_b]
    vectorizer = TfidfVectorizer()
    vectorizer.fit(cleaned_a + cleaned_b)
    matrix_a = vectorizer.transform(cleaned_a)
    matrix_b = vectorizer.transform(cleaned_b)
    return cosine_similarity(matrix_a, matrix_b).tolist()


def sentence_similarity_analysis(text: str) -> list[dict]:
    """For each sentence in `text`, find the most similar OTHER sentence in
    the same document. Useful for spotting repeated/boilerplate filler, not
    for detecting copied content from an external source.
    """
    sentences = split_sentences(text)
    if len(sentences) < 2:
        return [{"sentence": sentence, "most_similar_to": None, "similarity": 0.0} for sentence in sentences]

    matrix = _similarity_matrix(sentences, sentences)
    results = []
    for index, sentence in enumerate(sentences):
        best_index, best_score = None, 0.0
        for other_index, score in enumerate(matrix[index]):
            if other_index != index and score > best_score:
                best_index, best_score = other_index, score
        results.append({
            "sentence": sentence,
            "most_similar_to": sentences[best_index] if best_index is not None else None,
            "similarity": round(best_score * 100, 2),
        })
    return results


def sentence_level_corpus_matches(text: str, reference_documents: dict[str, str]) -> list[dict]:
    """For each sentence in `text`, find the best-matching sentence across
    every reference document, and which document it came from. This is the
    "highlight matching/plagiarized sentences" feature -- it looks outside
    the input document, unlike sentence_similarity_analysis().
    """
    sentences = split_sentences(text)
    if not sentences or not reference_documents:
        return [{"sentence": sentence, "source": None, "matched_sentence": None, "similarity": 0.0} for sentence in sentences]

    source_sentences: list[tuple[str, str]] = []  # (source name, sentence)
    for source_name, source_text in reference_documents.items():
        for sentence in split_sentences(source_text):
            source_sentences.append((source_name, sentence))
    if not source_sentences:
        return [{"sentence": sentence, "source": None, "matched_sentence": None, "similarity": 0.0} for sentence in sentences]

    reference_texts = [sentence for _source, sentence in source_sentences]
    matrix = _similarity_matrix(sentences, reference_texts)

    results = []
    for index, sentence in enumerate(sentences):
        best_index, best_score = max(enumerate(matrix[index]), key=lambda item: item[1], default=(None, 0.0))
        if best_index is None:
            results.append({"sentence": sentence, "source": None, "matched_sentence": None, "similarity": 0.0})
            continue
        source_name, matched_sentence = source_sentences[best_index]
        results.append({
            "sentence": sentence,
            "source": source_name,
            "matched_sentence": matched_sentence,
            "similarity": round(best_score * 100, 2),
        })
    return results
