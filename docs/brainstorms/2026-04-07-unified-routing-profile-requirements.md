---
date: 2026-04-07
topic: unified-routing-profile
---

# Unified Query Routing Profile

## Problem Frame

The retrieval pipeline has 8 tunable knobs (bm25_weight, hyde_enabled, hyde_weight, blend_weight, n_retrieve, preferred_sections, authority_boost, max_per_paper) but only bm25_weight adapts per-query. The remaining 7 are static within a mode. The query classifier (`_classify_query_bm25_weight`) already computes useful signals (entity count, compare/definitional patterns) but outputs a single float. This is the architectural enabler that gives HyDE gating, quality score gating, and adaptive reranker blend a clean home.

## Requirements

**Classifier Output**
- R1. Replace `_classify_query_bm25_weight(query) -> float` with `_classify_query(query) -> dict` returning a routing delta dict with keys: `bm25_weight`, `hyde_enabled`, `hyde_weight`, `blend_weight`, `n_retrieve_scale`, `section_prefs`, `authority_boost_scale`.
- R2. The delta dict uses multiplicative scales for numerical fields (e.g., `n_retrieve_scale: 0.5` means halve the mode's n_retrieve) and direct values for booleans/sets.
- R3. Fields omitted from the delta dict mean "use mode default". The classifier only sets fields where it has a signal.

**Layering**
- R4. Mode config is the base layer. Query classification produces a delta layer. The effective routing config for each query is: `mode_config | query_delta` (with multiplicative fields applied as multipliers).
- R5. The layering must happen early in the pipeline, before HyDE dispatch and retrieval, so all downstream stages see the effective config.

**Signals**
- R6. The classifier uses the same signals currently in `_classify_query_bm25_weight`: `_COMPARE_PATTERNS`, `_DEFINITIONAL_PATTERNS`, entity count from `_kb_entities`. No new signals in this feature.
- R7. The classifier is a pure function (no LLM calls, no DB queries). Must complete in <1ms.

**Observability**
- R8. Log the effective routing config (after layering) for each query. Include the raw classifier output and the mode base for debugging.

## Success Criteria

- All existing query-adaptive behavior (BM25 weight adjustment) continues to work identically.
- HyDE gating (idea 2) can be implemented by adding `hyde_enabled: False` to the delta dict for entity-heavy queries, with no changes to the layering infrastructure.
- Future features (quality gating, adaptive blend) can plug in by adding fields to the delta dict.

## Scope Boundaries

- Not adding new classification signals (temporal awareness, query decomposition). Only restructuring the existing classifier output.
- Not changing MODE_ROUTING structure or config.yaml schema. The delta is applied at runtime, not persisted.
- Not adding per-field precedence rules. All delta fields layer the same way (direct override for booleans/sets, multiplicative for numericals).

## Key Decisions

- **Layering over override**: Mode config remains the authoritative base. Query classifier produces deltas, not replacements. Rationale: modes are user-selected intent signals that should not be overridden by heuristics. The classifier refines within the mode's intent.
- **Multiplicative scales for numericals**: `n_retrieve_scale: 0.5` rather than `n_retrieve: 30`. Rationale: decouples the classifier from knowing the mode's absolute values. If a mode changes n_retrieve from 60 to 80, the classifier's "narrow query" signal still works.
- **Pure function classifier**: No LLM calls. The existing regex + entity set approach is fast enough and the user explicitly rejected LLM classification in a prior session.

## Dependencies / Assumptions

- `_kb_entities` set is available at query time (verified: built at startup).
- `_COMPARE_PATTERNS` and `_DEFINITIONAL_PATTERNS` regexes exist (verified: lines 228-243 in `src/api.py`).
- HyDE gating (idea 2) and paper-reference short-circuit (idea 7) will consume the routing profile but can be implemented independently.

## Outstanding Questions

### Deferred to Planning
- [Affects R2][Technical] Exact scale values for n_retrieve_scale by query type. Needs calibration.
- [Affects R5][Technical] Where exactly in the pipeline to compute and apply the layered config. Currently bm25_weight is read at RRF fusion time (line ~999); the new profile needs to be available earlier (before HyDE dispatch at line ~873).

## Next Steps

-> `/ce:plan` for structured implementation planning
