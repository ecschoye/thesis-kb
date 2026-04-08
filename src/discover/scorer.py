"""Relevance scoring for candidate papers."""

import math
import re
from datetime import datetime


def _keyword_score(title, abstract, keyword_terms):
    """Score based on how many KB-derived terms appear in the candidate."""
    if not keyword_terms:
        return 0.0
    text_lower = f"{title} {abstract}".lower()
    title_lower = title.lower()
    matched = 0
    for term in keyword_terms:
        if term in title_lower:
            matched += 2  # title matches count double
        elif term in text_lower:
            matched += 1
    # Normalize: max possible = 3 * len(keyword_terms)
    return min(matched / (len(keyword_terms) * 0.5), 1.0)


def _authority_score(citation_count, year, publication_types=None):
    """Score based on citations, recency, and venue type."""
    # Citation signal (log scale, capped)
    cite_score = min(math.log2(1 + citation_count) / 15.0, 1.0)

    # Recency bonus
    current_year = datetime.now().year
    recency = 0.0
    if year and year >= current_year - 1:
        recency = 0.15
    elif year and year >= current_year - 2:
        recency = 0.08

    # Venue bonus
    venue = 0.0
    if publication_types:
        if any(t in ("JournalArticle", "Conference") for t in publication_types):
            venue = 0.1

    return min(cite_score + recency + venue, 1.0)


def _embedding_score(abstract, embed_fn, anchor_embeddings):
    """Score by cosine similarity between candidate and thesis anchors."""
    if not abstract or not embed_fn or not anchor_embeddings:
        return None
    try:
        emb = embed_fn(abstract)
        # Cosine similarity against each anchor, take max
        best = 0.0
        for anchor in anchor_embeddings:
            dot = sum(a * b for a, b in zip(emb, anchor))
            norm_a = math.sqrt(sum(a * a for a in emb))
            norm_b = math.sqrt(sum(b * b for b in anchor))
            if norm_a > 0 and norm_b > 0:
                sim = dot / (norm_a * norm_b)
                best = max(best, sim)
        return best
    except Exception as e:
        return None


def score_candidates(candidates, profile, embed_fn=None, anchor_embeddings=None):
    """Score and sort candidates by relevance.

    With embeddings: 0.50 * embedding + 0.30 * keyword + 0.20 * authority
    Without:         0.60 * keyword + 0.40 * authority
    """
    scored = []
    use_embeddings = embed_fn is not None and anchor_embeddings

    for c in candidates:
        kw = _keyword_score(c.title, c.abstract, profile.keyword_terms)
        auth = _authority_score(c.citation_count, c.year)

        if use_embeddings:
            emb = _embedding_score(c.abstract, embed_fn, anchor_embeddings)
            if emb is not None:
                relevance = 0.50 * emb + 0.30 * kw + 0.20 * auth
            else:
                relevance = 0.60 * kw + 0.40 * auth
                emb = None
        else:
            relevance = 0.60 * kw + 0.40 * auth
            emb = None

        scored.append({
            "paper": c,
            "relevance": round(relevance, 4),
            "embedding_sim": round(emb, 4) if emb is not None else None,
            "keyword_score": round(kw, 4),
            "authority_score": round(auth, 4),
        })

    scored.sort(key=lambda x: x["relevance"], reverse=True)
    return scored


def compute_anchor_embeddings(queries, embed_fn):
    """Pre-compute embeddings for the profile's search queries."""
    anchors = []
    for q in queries:
        try:
            emb = embed_fn(q)
            anchors.append(emb)
        except Exception:
            continue
    return anchors
