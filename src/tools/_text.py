"""Text analysis utilities for thesis-kb tools.

Sentence splitting, readability metrics, AI-pattern detection.
All stdlib — no external dependencies.
"""

from __future__ import annotations

import re
from collections import Counter


# ---------------------------------------------------------------------------
# Sentence splitting (abbreviation-aware)
# ---------------------------------------------------------------------------

# Abbreviations that should NOT trigger sentence breaks
_ABBREVS = {
    "et al", "e.g", "i.e", "cf", "vs", "Fig", "Eq", "Tab", "Sec", "Ch",
    "Ref", "No", "Vol", "approx", "ca", "Dr", "Prof", "Mr", "Mrs", "Ms",
    "Inc", "Corp", "Ltd", "Jr", "Sr",
}
_ABBREV_RE = re.compile(
    r"\b(" + "|".join(re.escape(a) for a in _ABBREVS) + r")\.\s",
    re.IGNORECASE,
)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, respecting common abbreviations.

    Handles "et al.", "e.g.", "i.e.", "Fig.", decimal numbers, etc.
    """
    # Protect abbreviations with placeholder
    protected = text
    placeholders: list[tuple[str, str]] = []
    for m in _ABBREV_RE.finditer(text):
        orig = m.group(0)
        ph = f"__ABBR{len(placeholders)}__ "
        placeholders.append((ph.strip(), orig.strip()))
        protected = protected.replace(orig, ph, 1)

    # Protect decimal numbers like "3.14"
    protected = re.sub(r"(\d)\.(\d)", r"\1__DOT__\2", protected)

    # Split on sentence-ending punctuation followed by space + uppercase or end
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\[])", protected)

    sentences = []
    for part in parts:
        s = part.strip()
        if not s:
            continue
        # Restore placeholders
        for ph, orig in placeholders:
            s = s.replace(ph, orig)
        s = s.replace("__DOT__", ".")
        sentences.append(s)
    return sentences


# ---------------------------------------------------------------------------
# Readability (Flesch-Kincaid)
# ---------------------------------------------------------------------------

def _count_syllables(word: str) -> int:
    """Approximate syllable count for English word."""
    word = word.lower().strip(".,;:!?\"'()-")
    if not word:
        return 0
    count = 0
    vowels = "aeiouy"
    prev_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    # Adjust for silent e
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def flesch_kincaid(text: str) -> dict:
    """Compute Flesch-Kincaid readability metrics.

    Returns {grade_level, reading_ease, words, sentences, syllables}.
    """
    sentences = split_sentences(text)
    words = text.split()
    n_sentences = max(1, len(sentences))
    n_words = max(1, len(words))
    n_syllables = sum(_count_syllables(w) for w in words)

    avg_sentence_len = n_words / n_sentences
    avg_syllables_per_word = n_syllables / n_words

    grade = 0.39 * avg_sentence_len + 11.8 * avg_syllables_per_word - 15.59
    ease = 206.835 - 1.015 * avg_sentence_len - 84.6 * avg_syllables_per_word

    return {
        "grade_level": round(grade, 1),
        "reading_ease": round(ease, 1),
        "words": n_words,
        "sentences": n_sentences,
        "syllables": n_syllables,
    }


# ---------------------------------------------------------------------------
# Word and paragraph counting
# ---------------------------------------------------------------------------

def word_count(text: str) -> int:
    return len(text.split())


def count_paragraphs(text: str) -> int:
    """Count paragraphs (separated by blank lines or \\par)."""
    paras = re.split(r"\n\s*\n|\\par\b", text)
    return sum(1 for p in paras if p.strip())


# ---------------------------------------------------------------------------
# AI-writing pattern detection
# ---------------------------------------------------------------------------

# Patterns characteristic of AI-generated text, grouped by category
AI_PATTERNS: dict[str, list[str]] = {
    "filler_phrases": [
        r"\bit is (?:important|worth|crucial|essential) to note\b",
        r"\bit should be noted\b",
        r"\bin the realm of\b",
        r"\bin the context of\b",
        r"\bin today's (?:rapidly evolving|fast-paced|modern)\b",
        r"\bpaving the way\b",
        r"\ba testament to\b",
        r"\bit is (?:important|crucial|essential) to (?:understand|recognize|acknowledge)\b",
    ],
    "overused_verbs": [
        r"\bdelve(?:s|d)?\b",
        r"\bleverage(?:s|d)?\b",
        r"\bharness(?:es|ed)?\b",
        r"\bunlock(?:s|ed)?\b",
        r"\bunderpin(?:s|ned)?\b",
        r"\bpivotal\b",
        r"\bseamless(?:ly)?\b",
        r"\bholistic(?:ally)?\b",
        r"\brobust(?:ly|ness)?\b",
    ],
    "superlatives_without_evidence": [
        r"\bgroundbreaking\b",
        r"\brevolutionary\b",
        r"\bunprecedented\b",
        r"\bcutting-edge\b",
        r"\bstate-of-the-art\b(?!\s*\()",  # ok if followed by citation
        r"\btransformative\b",
    ],
    "connector_overuse": [
        r"\bfurthermore\b",
        r"\bmoreover\b",
        r"\badditionally\b",
        r"\bconsequently\b",
        r"\bnevertheless\b",
    ],
}

_COMPILED_PATTERNS: dict[str, list[re.Pattern]] = {
    cat: [re.compile(p, re.IGNORECASE) for p in patterns]
    for cat, patterns in AI_PATTERNS.items()
}


def detect_ai_phrases(text: str) -> list[dict]:
    """Detect AI-writing patterns in text.

    Returns list of {category, pattern, match, line, position}.
    """
    lines = text.split("\n")
    results = []
    offset = 0
    for lineno, line in enumerate(lines, 1):
        for cat, patterns in _COMPILED_PATTERNS.items():
            for pat in patterns:
                for m in pat.finditer(line):
                    results.append({
                        "category": cat,
                        "pattern": pat.pattern,
                        "match": m.group(0),
                        "line": lineno,
                        "position": offset + m.start(),
                        "context": line.strip(),
                    })
        offset += len(line) + 1
    return results


def ai_pattern_density(text: str) -> float:
    """Return AI-pattern matches per 1000 words."""
    n_words = max(1, word_count(text))
    n_matches = len(detect_ai_phrases(text))
    return round(n_matches / n_words * 1000, 2)
