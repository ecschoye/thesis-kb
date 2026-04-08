"""Deduplicate candidates against the existing KB and each other."""

from collections import defaultdict
from difflib import SequenceMatcher

from src.acquire.zotero import _norm


def _title_sim(a, b):
    return SequenceMatcher(None, a, b).ratio()


def _build_trigram_index(title_norms):
    """Build a trigram → title set index for fast pre-filtering."""
    index = defaultdict(set)
    for t in title_norms:
        words = t.split()
        for w in words[:6]:  # first 6 words
            if len(w) >= 3:
                index[w].add(t)
    return index


def _fast_fuzzy_match(candidate_norm, trigram_index, all_norms, threshold=0.85):
    """Check if candidate matches any existing title using trigram pre-filter.

    Only runs expensive SequenceMatcher on titles that share at least one word.
    """
    words = candidate_norm.split()
    check_set = set()
    for w in words[:6]:
        if len(w) >= 3 and w in trigram_index:
            check_set.update(trigram_index[w])

    # If no trigram overlap at all, also check exact containment
    if not check_set:
        # Very short titles could slip through — check exact match
        if candidate_norm in all_norms:
            return True
        return False

    for existing in check_set:
        if _title_sim(candidate_norm, existing) >= threshold:
            return True
    return False


def deduplicate(candidates, profile, threshold=0.85):
    """Remove candidates already in KB + cross-source duplicates.

    Returns (new_papers, stats_dict).
    """
    # Pre-build trigram index for fast fuzzy matching
    kb_trigrams = _build_trigram_index(profile.exclude_title_norms)

    seen_ids = set()
    seen_norms = set()
    seen_trigrams = defaultdict(set)  # for cross-source dedup
    new_papers = []
    stats = {"exact_id": 0, "fuzzy_title": 0, "cross_source": 0, "kept": 0}

    for c in candidates:
        # Stage 1: exact ID match against KB
        if c.arxiv_id and c.arxiv_id in profile.exclude_arxiv_ids:
            stats["exact_id"] += 1
            continue
        if c.doi and c.doi.lower() in profile.exclude_dois:
            stats["exact_id"] += 1
            continue

        # Stage 2: fuzzy title match against KB (with trigram pre-filter)
        c_norm = _norm(c.title) if c.title else ""
        if not c_norm:
            continue

        if _fast_fuzzy_match(c_norm, kb_trigrams, profile.exclude_title_norms, threshold):
            stats["fuzzy_title"] += 1
            continue

        # Stage 3: cross-source dedup within candidates
        dup_key = c.arxiv_id or (c.doi.lower() if c.doi else None)
        if dup_key and dup_key in seen_ids:
            stats["cross_source"] += 1
            continue

        if _fast_fuzzy_match(c_norm, seen_trigrams, seen_norms, threshold):
            stats["cross_source"] += 1
            continue

        # Accept
        if dup_key:
            seen_ids.add(dup_key)
        seen_norms.add(c_norm)
        # Add to cross-source trigram index
        for w in c_norm.split()[:6]:
            if len(w) >= 3:
                seen_trigrams[w].add(c_norm)
        new_papers.append(c)
        stats["kept"] += 1

    return new_papers, stats
