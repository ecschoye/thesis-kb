"""Language analysis tools for thesis-kb."""

from __future__ import annotations

import re

from . import _latex as L
from . import _text as T


def register(mcp):

    @mcp.tool()
    def detect_ai_patterns(tex_content: str) -> dict:
        """Flag phrases and patterns characteristic of AI-generated text.

        Detects: filler phrases ("it is important to note"), overused verbs
        ("delve", "leverage"), superlatives without evidence, and connector
        overuse ("furthermore", "moreover"). Reports density per 1000 words.

        Args:
            tex_content: LaTeX source text to analyze.
        """
        plain = L.strip_latex(tex_content)
        matches = T.detect_ai_phrases(plain)
        density = T.ai_pattern_density(plain)
        n_words = T.word_count(plain)

        # Group by category
        from collections import Counter
        by_category = Counter(m["category"] for m in matches)

        return {
            "total_matches": len(matches),
            "density_per_1000_words": density,
            "word_count": n_words,
            "by_category": dict(by_category),
            "matches": matches[:30],  # Limit output
        }

    @mcp.tool()
    def check_first_person(tex_content: str) -> dict:
        """Find first-person pronoun usage (supervisor rule 1).

        Scans for I, we, our, us, my outside of quoted text and math mode.

        Args:
            tex_content: LaTeX source text.
        """
        lines = tex_content.split("\n")
        matches = []
        PRONOUNS = re.compile(r"\b(I|we|our|us|my)\b", re.IGNORECASE)

        for lineno, line in enumerate(lines, 1):
            stripped = line.strip()
            # Skip comments
            if stripped.startswith("%"):
                continue
            # Skip lines that are entirely in math mode
            if stripped.startswith("$") or stripped.startswith("\\["):
                continue

            for m in PRONOUNS.finditer(stripped):
                word = m.group(0)
                # Skip "I" if it's likely an index variable in math
                if word == "I" and ("$" in stripped[:m.start()] and "$" in stripped[m.end():]):
                    continue
                matches.append({
                    "pronoun": word,
                    "line": lineno,
                    "context": stripped[:120],
                })

        return {
            "total_matches": len(matches),
            "matches": matches,
        }

    @mcp.tool()
    def check_hedging_strength(tex_content: str) -> dict:
        """Analyze claim strength: flag overclaiming and excessive hedging.

        Classifies sentences as:
        - strong_claim: no hedge, no citation (potential overclaim)
        - hedged_claim: contains hedge words
        - cited_claim: has \\cite{}
        - neutral: definition/description, not a claim

        Args:
            tex_content: LaTeX source text.
        """
        plain = L.strip_latex(tex_content)
        sentences = T.split_sentences(plain)

        hedge_words = {"may", "might", "could", "potentially", "suggests", "appears",
                       "seems", "possibly", "arguably", "likely", "presumably",
                       "it is possible", "to some extent"}
        claim_indicators = {"show", "demonstrate", "prove", "achieve", "outperform",
                            "improve", "result", "find", "observe", "reveal",
                            "indicate", "confirm", "establish", "better", "faster",
                            "higher", "lower", "superior", "efficient"}

        classifications = {
            "strong_claim": [],
            "hedged_claim": [],
            "cited_claim": [],
            "neutral": [],
        }

        for sent in sentences:
            sent_lower = sent.lower()
            has_cite = "cite" in sent_lower or "[" in sent  # rough check for citation markers
            has_hedge = any(h in sent_lower for h in hedge_words)
            has_claim = any(c in sent_lower for c in claim_indicators)

            if has_cite:
                classifications["cited_claim"].append(sent)
            elif has_claim and not has_hedge:
                classifications["strong_claim"].append(sent)
            elif has_hedge:
                classifications["hedged_claim"].append(sent)
            else:
                classifications["neutral"].append(sent)

        total = max(1, len(sentences))
        return {
            "total_sentences": len(sentences),
            "strong_claims": len(classifications["strong_claim"]),
            "hedged_claims": len(classifications["hedged_claim"]),
            "cited_claims": len(classifications["cited_claim"]),
            "neutral": len(classifications["neutral"]),
            "strong_claim_pct": round(len(classifications["strong_claim"]) / total * 100, 1),
            "hedged_claim_pct": round(len(classifications["hedged_claim"]) / total * 100, 1),
            "potential_overclaims": classifications["strong_claim"][:10],
            "heavily_hedged": classifications["hedged_claim"][:10],
        }
