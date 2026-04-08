---
date: 2026-04-07
topic: hyde-gating
---

# Query-Adaptive HyDE Gating

## Problem Frame

HyDE (Hypothetical Document Embedding) generates a hypothetical answer for every query and embeds it as an additional retrieval vector. This costs ~500ms (LLM round-trip) and can degrade precision for entity-heavy factual queries where the hypothetical answer hallucates wrong details (invented numbers, wrong method names). The current HyDE prompt is also generic regardless of query mode, producing textbook-style responses even when the user is looking for experimental results or limitations.

## Requirements

**Gating**
- R1. Disable HyDE for queries where it is likely to hurt: (a) 2+ KB entities detected (entity-heavy factual lookups), (b) short queries (<8 tokens) with a KB entity (precise lookups), (c) queries that will be short-circuited by the paper-reference feature.
- R2. Always enable HyDE for: (a) conceptual/definitional queries (no KB entities, broad phrasing), (b) synthesis queries (outline, survey modes), (c) queries where the expansion LLM would benefit from a semantic anchor.
- R3. The gating decision must be made before the HyDE LLM call is dispatched (line 900 in `src/api.py`), not after.

**Mode-Aware Prompting**
- R4. When HyDE is enabled, condition the generation prompt on the active mode's `preferred_sections`:
  - `background` mode: textbook-style explanatory passage
  - `check`/`review` mode: findings-style passage with specific results and comparisons
  - `gaps` mode: limitations and future work style passage
  - `draft`/`outline`/default: current generic academic style (no change)
- R5. The mode-aware prompt must be a simple template swap, not a separate LLM call.

**Observability**
- R6. Log whether HyDE was enabled or disabled for each query, and the reason (entity count, mode, query length).

## Success Criteria

- Entity-heavy factual queries ("what mAP does RENet achieve on DSEC?") skip HyDE, saving ~500ms latency.
- Conceptual queries still get HyDE and benefit from the semantic anchor.
- Mode-aware prompts produce hypothetical passages that better match the style of target nuggets (background vs. results vs. limitations).

## Scope Boundaries

- Not adding per-query HyDE weight tuning (hyde_weight stays static when HyDE is enabled). That belongs in the unified routing profile (idea 3).
- Not evaluating HyDE quality empirically in this feature. Gating is based on structural query signals.
- Not changing the HyDE embedding path (`_cached_embed_raw`). Only the gating decision and prompt content change.

## Key Decisions

- **Gating uses existing classifier signals**: Reuse entity count and pattern detection from `_classify_query_bm25_weight` rather than adding a new classifier. Requires moving classification earlier in the pipeline (before HyDE dispatch).
- **Mode-aware prompting**: Use `preferred_sections` from MODE_ROUTING to select a prompt template. This is a natural extension of the existing mode infrastructure.

## Dependencies / Assumptions

- Depends on `_classify_query_bm25_weight` or its signals being available before HyDE dispatch. Currently the classifier runs after retrieval. The classification logic must be extracted or moved earlier.
- Entity set (`_kb_entities`) is available at query time (verified: built at startup, line ~457).

## Outstanding Questions

### Deferred to Planning
- [Affects R1][Technical] Exact entity_hits threshold for gating. 2+ is the initial proposal but may need calibration.
- [Affects R4][Technical] Prompt template wording for each mode. Should be concise and testable.

## Next Steps

-> `/ce:plan` for structured implementation planning
