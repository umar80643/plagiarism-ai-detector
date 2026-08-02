"""Human-readable reasons behind an AI-detection prediction.

Kept independent of any particular backend (heuristic, sklearn, transformer)
deliberately: "why" a piece of text looks AI-generated is a property of the
text's surface statistics, not of which classifier happened to make the
final call, and this is honest about what's actually computed.

Scope note: this covers the three signals that can be computed directly from
the text (lexical diversity, sentence-length uniformity, internal
repetition). "Excessive predictability" and "unusual fluency" -- also
requested -- need a language-model perplexity estimate (how surprised a
model is by each next word), which isn't implemented here; see the
"Future improvements" note in README for what that would take. Better to
list three real, computed signals than fake five.
"""
from __future__ import annotations

from config import DEFAULT_AI_DETECTOR_SETTINGS
from utils.similarity import sentence_similarity_analysis
from utils.text_analysis import analyze_text


def explain_ai_signals(text: str, settings=DEFAULT_AI_DETECTOR_SETTINGS) -> list[str]:
    stats = analyze_text(text)
    reasons: list[str] = []

    if stats["lexical_diversity"] < settings.low_lexical_diversity:
        reasons.append(
            f"low lexical diversity ({stats['lexical_diversity']:.2f}, vocabulary is unusually repetitive for this length)"
        )
    if stats["burstiness"] < settings.low_burstiness:
        reasons.append(
            f"highly uniform sentence structure (sentence-length variance is {stats['burstiness']:.2f}, unusually low)"
        )

    repeats = [item for item in sentence_similarity_analysis(text) if item["similarity"] > 70 and item["most_similar_to"]]
    if repeats:
        reasons.append(f"repetitive phrasing detected in {len(repeats)} sentence(s)")

    return reasons
