---
date: 2026-04-07
topic: paper-reference-short-circuit
---

# Paper-Reference Short-Circuit

## Problem Frame

When a user query references a specific paper (via `\cite{key}`, arXiv ID, or paper_id), the system runs the full retrieval pipeline: expand, embed, retrieve (vector + BM25), RRF fusion, rerank. This is wasteful because the user already identified the exact paper. The infrastructure to resolve paper references and fetch nuggets directly exists (`_extract_cite_keys`, `_resolve_cite_to_paper_id`, `get_paper_nuggets`) but the pipeline never short-circuits.

## Requirements

**Detection**
- R1. Detect paper references in three forms: `\cite{key}` patterns, arXiv IDs (`YYMM.NNNNN`), and paper_id patterns (`YYMM_NNNNN`).
- R2. Resolve each detected reference to a `paper_id` using existing infrastructure (`_resolve_cite_to_paper_id` for cite keys, SQLite lookup for arXiv IDs and paper_ids).

**Short-Circuit Activation**
- R3. Short-circuit activates when: (a) all detected references resolve to valid paper_ids, (b) the query contains no cross-paper signals (comparison patterns, "other", "alternative", "between X and Y"), and (c) the query intent is single-paper-scoped (summarize, explain, list, describe).
- R4. When short-circuit does not activate (cross-paper intent detected), fall back to the normal retrieval pipeline with the resolved papers added to `pinned_papers` (current behavior).

**Retrieval**
- R5. In short-circuit mode, fetch nuggets directly from SQLite for the resolved paper(s), filtered by the active mode's `allowed_types` and `preferred_sections` if set.
- R6. Apply the mode's `max_per_paper` limit. If no mode is active, use a reasonable default (e.g., 20 nuggets).

**Response**
- R7. The LLM still receives the nuggets as context and generates a streamed SSE response, identical to normal flow. Only the retrieval path changes.
- R8. The SSE metadata event should indicate that short-circuit retrieval was used (for debugging/logging).

## Success Criteria

- Paper-specific queries (`summarize \cite{Zhu_2019}`, `what methods does 2401.17151 use?`) return results in <200ms instead of multi-second retrieval.
- Cross-paper queries mentioning a paper still get full retrieval with the paper pinned.
- No regression in result quality for non-paper-reference queries.

## Scope Boundaries

- Not adding a new `/paper` or `/cite` slash command mode. This is automatic detection within the existing pipeline.
- Not changing the LLM prompt or response format. Only the retrieval path is affected.
- Not handling partial title matches (e.g., "the EV-FlowNet paper"). Only explicit identifiers.

## Key Decisions

- **Cite-only short-circuit**: Activate only when the query is purely paper-scoped. Cross-paper queries fall back to normal retrieval with pinning. Rationale: avoids the edge case where short-circuit misses cross-paper context the user actually wants.
- **Three detection patterns**: `\cite{}`, arXiv IDs, paper_ids. Rationale: all three are common in thesis-writing workflows and have low false-positive risk due to their distinctive formats.

## Dependencies / Assumptions

- `_resolve_cite_to_paper_id` and the bib lookup are loaded at startup (verified: lines 343-387 in `src/api.py`).
- The compare pattern regex `_COMPARE_PATTERNS` (already exists) can be reused for cross-paper signal detection.

## Outstanding Questions

### Deferred to Planning
- [Affects R5][Technical] Should nuggets in short-circuit mode be ranked by `thesis_relevance` or `overall_score`, or returned in section order?
- [Affects R3][Technical] Exact regex for arXiv ID and paper_id detection. Should handle both `2401.17151` and `2401_17151` forms in query text.

## Next Steps

-> `/ce:plan` for structured implementation planning
