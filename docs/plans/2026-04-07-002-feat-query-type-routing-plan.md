---
title: "feat: Query-type routing — short-circuit, unified profile, HyDE gating"
type: feat
status: completed
date: 2026-04-07
origin:
  - docs/brainstorms/2026-04-07-paper-reference-short-circuit-requirements.md
  - docs/brainstorms/2026-04-07-unified-routing-profile-requirements.md
  - docs/brainstorms/2026-04-07-hyde-gating-requirements.md
---

# feat: Query-type routing — short-circuit, unified profile, HyDE gating

## Overview

Three retrieval improvements to `src/api.py` that make the query pipeline adapt to query characteristics:

1. **Paper-Reference Short-Circuit** — skip full retrieval for paper-specific queries
2. **Unified Query Routing Profile** — replace single-float classifier with a delta-dict that tunes all retrieval knobs
3. **Query-Adaptive HyDE Gating** — disable HyDE for entity-heavy queries, mode-aware prompting when enabled

## Problem Frame

The retrieval pipeline has 8 tunable knobs but only `bm25_weight` adapts per-query. Paper-specific queries waste seconds on expand/embed/retrieve when a DB lookup would suffice. HyDE generates hypothetical answers for every query, including factual lookups where hallucinated details hurt precision. (See origin docs for full context.)

## Requirements Trace

From paper-reference short-circuit requirements:
- R1-SC. Detect paper references in three forms: `\cite{key}`, arXiv IDs, paper_id patterns
- R2-SC. Resolve references to paper_ids using existing infrastructure
- R3-SC. Short-circuit when exactly one paper resolved and no cross-paper signals detected
- R4-SC. Fall back to normal pipeline with pinning when cross-paper intent detected
- R5-SC. Fetch nuggets from SQLite, filtered by mode's allowed_types/preferred_sections
- R6-SC. Apply max_per_paper limit
- R7-SC. LLM still receives nuggets and streams SSE response
- R8-SC. SSE metadata indicates short-circuit was used

From unified routing profile requirements:
- R1-UR. Replace `_classify_query_bm25_weight() -> float` with `_classify_query() -> dict`
- R2-UR. Delta dict uses multiplicative scales for numericals, direct values for booleans/sets
- R3-UR. Omitted fields mean "use mode default"
- R4-UR. Mode config is base layer, query delta layers on top
- R5-UR. Layering happens before HyDE dispatch and retrieval
- R6-UR. Same signals as current classifier (no new signals)
- R7-UR. Pure function, <1ms
- R8-UR. Log effective routing config per query

From HyDE gating requirements:
- R1-HG. Disable HyDE for entity-heavy (2+ entities), short precise queries (<8 tokens + entity)
- R2-HG. Always enable HyDE for conceptual/synthesis queries
- R3-HG. Gating decision before HyDE LLM call dispatch
- R4-HG. Mode-aware prompt templates (background/results/limitations/generic)
- R5-HG. Template swap, not separate LLM call
- R6-HG. Log HyDE gating decision and reason

## Scope Boundaries

- No new classification signals (temporal, decomposition). Only restructuring existing classifier output. (Note: HyDE gating introduces query token count as a minor new signal for the short-query check; this is lightweight enough to not warrant a separate classifier.)
- No per-query HyDE weight tuning (hyde_weight stays static when HyDE is on).
- No partial title matching for short-circuit. Only explicit identifiers.
- No changes to MODE_ROUTING structure or config.yaml schema.
- No test suite (repo has none; verification is manual + import check).

## Context & Research

### Relevant Code and Patterns

- `src/api.py:245-273` — `_classify_query_bm25_weight()`, the function being replaced
- `src/api.py:47-97` — `_DEFAULT_MODE_ROUTING`, the mode config consumed downstream
- `src/api.py:331-340` — `_extract_cite_keys()`, cite key extraction (only `\cite{}` syntax)
- `src/api.py:343-405` — `_resolve_cite_to_paper_id()`, resolves cite keys to paper_ids
- `src/api.py:1415-1434` — Cite extraction + pinning block in chat handler (BEFORE `_run_retrieval`)
- `src/api.py:872-920` — HyDE generation + embedding (INSIDE `_run_retrieval`, Step 1b)
- `src/api.py:961-1008` — RRF fusion where bm25_weight is applied (INSIDE `_run_retrieval`, Step 4)
- `src/api.py:1138-1154` — Reranker call where blend_weight is passed
- `src/api.py:1219-1262` — Pinned papers injection (post-RRF, fixed score 1.0)
- `src/api.py:200-242` — `_build_entity_set()`, KB-derived entity extraction at startup
- `src/api.py:228-243` — `_COMPARE_PATTERNS`, `_DEFINITIONAL_PATTERNS` regexes
- `src/rerank.py:36-119` — `rerank_nuggets()` with `blend_weight` parameter

### Key Architecture Finding

`_classify_query_bm25_weight` is called INSIDE `_run_retrieval` at line 998, but HyDE dispatch happens at line 900 (also inside `_run_retrieval`). For the unified profile to gate HyDE, classification must move earlier — either to the chat handler (before calling `_run_retrieval`) or to the top of `_run_retrieval` before HyDE dispatch.

**Decision:** Run classification at the top of `_run_retrieval`, before HyDE dispatch. This keeps the routing logic self-contained within the retrieval function rather than splitting it across the chat handler. The short-circuit check stays in the chat handler (it needs to skip `_run_retrieval` entirely).

## Key Technical Decisions

- **Classification inside `_run_retrieval`**: The classifier runs at the top of `_run_retrieval` (before Step 1), not in the chat handler. The routing profile dict is computed once and consumed by all downstream steps. Rationale: keeps retrieval logic self-contained. (See origin: unified-routing-profile-requirements.md)
- **Short-circuit in chat handler**: Paper-reference detection and short-circuit decision happen in the chat handler, before calling `_run_retrieval`. If short-circuit activates, `_run_retrieval` is never called. Rationale: short-circuit skips the entire retrieval function. (See origin: paper-reference-short-circuit-requirements.md)
- **Layering precedence**: `request params > classifier delta > mode config > global defaults`. Non-None classifier fields replace mode values. Multiplicative scales (e.g., `n_retrieve_scale: 0.5`) multiply the mode's absolute value. (See origin: unified-routing-profile-requirements.md)
- **HyDE prompt templates**: Four templates keyed by section category: `background` (textbook-style), `results` (findings with specifics), `limitations` (future work style), `generic` (current prompt). Selected by majority section in `preferred_sections` after layering. (See origin: hyde-gating-requirements.md)
- **Fallback on short-circuit failure**: If resolved paper_ids return zero nuggets (paper not in KB or empty), fall back to full `_run_retrieval` pipeline. Log the fallback.

## Open Questions

### Resolved During Planning

- **Where does classification run?** Inside `_run_retrieval`, at the top before HyDE dispatch. Not in the chat handler.
- **Bare arXiv ID regex?** Pattern: `r'\b(\d{4}\.\d{4,5})\b'` for arXiv IDs, `r'\b(\d{4}_\d{4,5})\b'` for paper_ids. Low false-positive risk due to distinctive format.
- **Nugget ordering in short-circuit?** Sort by `thesis_relevance` DESC, then `overall_score` DESC (if non-null), then `section` order matching mode's preferred_sections.
- **Section_prefs merge semantics?** Classifier non-None replaces mode value (not union/intersection). Classifier only sets this when it has a strong query-content signal.

### Deferred to Implementation

- Exact entity_hits threshold for HyDE gating (start with 2, calibrate if needed)
- HyDE prompt template wording (start with simple variants, iterate based on output quality)
- ~~Whether `n_retrieve_scale` should clamp to a range~~ **Resolved:** Clamp to [0.3, 3.0] in `_build_effective_config` from day one. Cost is one line; risk of not clamping is silent retrieval failure.

## High-Level Technical Design

> *This illustrates the intended approach and is directional guidance for review, not implementation specification.*

```
Chat Handler (src/api.py /chat endpoint):
  1. Extract paper references (cite keys + arXiv IDs + paper_ids)
  2. Resolve to paper_ids
  3. Check cross-paper signals (_COMPARE_PATTERNS, etc.)
  4. If single-paper-scoped AND all resolved:
       -> SHORT-CIRCUIT: fetch nuggets from SQLite, format, stream to LLM
  5. Else:
       -> Add resolved papers to pinned_papers (existing behavior)
       -> Call _run_retrieval(query, mode, routing_profile=None, ...)

_run_retrieval:
  Step 0 (NEW): Classify query -> routing_profile dict
         Layer: mode_config | routing_profile -> effective_config
  Step 1a: Expand query (existing)
  Step 1b: HyDE — gated by effective_config["hyde_enabled"]
           Prompt selected by effective_config["preferred_sections"]
  Step 2: Embed variants (existing)
  Step 3: Retrieve vector + BM25 (n_retrieve from effective_config)
  Step 4: RRF fusion (bm25_weight from effective_config)
  Step 5: Boosts (authority_boost from effective_config)
  Step 6: Rerank (blend_weight from effective_config)
  Step 7: Select top nuggets (existing)
```

## Implementation Units

- [x] **Unit 1: Paper-Reference Detection & Short-Circuit**

**Goal:** Detect paper references in queries and skip full retrieval for single-paper queries.

**Requirements:** R1-SC, R2-SC, R3-SC, R4-SC, R5-SC, R6-SC, R7-SC, R8-SC

**Dependencies:** None (independent of other units)

**Files:**
- Modify: `src/api.py`

**Approach:**
- Extend `_extract_cite_keys` (or add a thin wrapper) to also detect bare arXiv IDs (`\b(?:1[0-9]|2[0-4])\d{2}\.\d{4,5}\b`) and paper_id patterns (`\b\d{4}_\d{4,5}\b`). Deduplicate resolved paper_ids (arXiv `2401.17151` and paper_id `2401_17151` resolve to the same paper). Only treat matched patterns as paper references if they actually resolve in the SQLite papers table.
- In the chat handler, before the existing cite-pinning block (~line 1415), for ALL modes (not restricted to the cite-pinning whitelist):
  - Extract and resolve paper references from `last_msg`
  - Check for cross-paper signals: `_COMPARE_PATTERNS.search(last_msg)`
  - If exactly ONE paper resolved AND no cross-paper signals → short-circuit path
  - If multiple papers resolved (with or without cross-paper signals) → fall through to pinning + `_run_retrieval`
  - If any refs unresolved → fall through to `_run_retrieval` (no pinning for unresolved refs)
- Short-circuit path:
  - Query SQLite for nuggets from the resolved paper_id, filtered by mode's `allowed_types`.
  - Use `preferred_sections` as a sort-priority boost (matching the normal pipeline's soft boost behavior), NOT as a hard filter. This ensures consistent behavior if the query falls back to full retrieval.
  - Sort by: preferred_section match first, then `thesis_relevance` DESC, then `overall_score` DESC (if non-null).
  - Cap at mode's `max_per_paper` (or 20 if no mode set).
  - Enrich each nugget with paper metadata from SQLite: `paper_title`, `paper_year`, `paper_authors`, `arxiv_id`. Resolve `bibtex_key` via `_resolve_bibtex_key`. Set `rrf_score: 1.0`, `overlap_count: 0`, `matched_queries: []`.
  - Return `(top_nuggets, [])` — the empty variants list signals short-circuit to the SSE streaming code. Falls through to existing LLM context assembly and streaming code (no duplication).
  - Include `"retrieval_mode": "short-circuit"` in the SSE sources event.
- Fallback: if short-circuit yields zero nuggets, log warning, add paper to `pinned_papers`, and fall through to `_run_retrieval`.
- Logging: INFO level `"Short-circuit: paper_id=%s, %d nuggets returned"` or `"Short-circuit fallback: %s"` with reason.

**Patterns to follow:**
- Existing cite-pinning block at lines 1415-1434 for detection/resolution flow
- Pinned papers injection at lines 1219-1262 for nugget formatting (same dict structure)
- SSE metadata event pattern at line 1557-1581

**Test scenarios:**
- Happy path: query `"summarize \cite{Zhu_2019}"` → detects cite key, resolves to paper_id, returns enriched nuggets (with bibtex_key, paper_title, etc.), SSE metadata shows `retrieval_mode: short-circuit`
- Happy path: query `"what methods does 2401.17151 use?"` → detects bare arXiv ID, resolves, short-circuits
- Happy path: query `"explain 2401_17151"` → detects paper_id pattern, resolves, short-circuits
- Happy path: short-circuit nuggets have all fields required by SSE sources payload (bibtex_key, paper_title, paper_year, paper_authors, arxiv_id, rrf_score, overlap_count, matched_queries)
- Edge case: query `"compare \cite{Zhu_2019} to other event representations"` → detects cross-paper signal, falls back to pinning + full retrieval
- Edge case: query `"2401.17151 vs 2312.00001"` → two arXiv IDs + comparison → falls back to pinning
- Edge case: query `"\cite{NonExistent_2099}"` → cite key does not resolve → no short-circuit, proceed to full retrieval (no pinning either)
- Edge case: query `"\cite{Zhu_2019}"` resolves but paper has zero nuggets in KB → fall back to full pipeline with that paper pinned
- Edge case: query with no paper references → no change to existing behavior
- Edge case: query `"summarize \cite{Zhu_2019} and \cite{Wang_2024}"` (two papers, no comparison) → does NOT short-circuit (multi-paper), falls back to pinning + full retrieval
- Edge case: query `"mAP improved to 0.9845 on the benchmark"` → no arXiv pattern match (4-digit.4-5-digit with plausible year prefix required)
- Edge case: query in `/gaps` mode (not in original cite-pinning whitelist) with `\cite{Zhu_2019}` → short-circuit still activates (not mode-restricted)

**Verification:**
- Paper-reference queries return results in <200ms (vs multi-second with full pipeline)
- Cross-paper queries still get full retrieval with pinning
- Non-reference queries are unaffected

---

- [x] **Unit 2: Unified Query Routing Profile**

**Goal:** Replace the single-float classifier with a delta-dict classifier, and add layering logic that merges it with mode config.

**Requirements:** R1-UR, R2-UR, R3-UR, R4-UR, R5-UR, R6-UR, R7-UR, R8-UR

**Dependencies:** None (can be built in parallel with Unit 1, but Unit 3 depends on this)

**Files:**
- Modify: `src/api.py`

**Approach:**
- Rename `_classify_query_bm25_weight` to `_classify_query`. Return a dict with keys: `bm25_weight` (float), `hyde_enabled` (bool|None), `n_retrieve_scale` (float|None), `blend_weight` (float|None), `section_prefs` (set|None), `authority_boost_scale` (float|None). None means "defer to mode config."
- Initially, the function returns the same bm25_weight calculation as before. All other fields are None. This preserves existing behavior exactly.
- Add `_build_effective_config(mode: str, query_delta: dict) -> dict`:
  - Start with `dict(MODE_ROUTING.get(mode, {}))` as base
  - For each non-None field in query_delta:
    - `bm25_weight`: direct override (it has no mode equivalent)
    - `hyde_enabled`: direct override (currently global, not per-mode)
    - `n_retrieve_scale`: multiply mode's `n_retrieve` and round to int
    - `blend_weight`: direct override
    - `section_prefs`: replace mode's `preferred_sections`
    - `authority_boost_scale`: multiply mode's `authority_boost`
  - Return the merged dict
- At the top of `_run_retrieval` (before Step 1a, line ~730), call `_classify_query(query)` and `_build_effective_config(mode, delta)`. Store as `effective`.
- Replace all downstream references to mode routing and static config with `effective` dict reads:
  - `hyde_enabled` from `effective` instead of `_retrieval_cfg`
  - `bm25_weight` from `effective` instead of calling `_classify_query_bm25_weight`
  - `rerank_weight` parameter defaults to `effective.get("blend_weight", rerank_weight)`
  - `effective_n_retrieve` from `effective.get("n_retrieve", ...)`
- Log effective config at debug level.

**Patterns to follow:**
- `_load_retrieval_config` at line 101 for mode config merging pattern
- `_classify_query_bm25_weight` at line 245 for classifier structure

**Test scenarios:**
- Happy path: query with no special signals → effective config equals mode config + default bm25_weight (1.0)
- Happy path: comparison query → effective config has `bm25_weight: 1.5` (same as current behavior)
- Happy path: entity-heavy query → effective config has `bm25_weight: 1.5+`
- Edge case: mode with no config entry (unknown mode string) → falls back to empty dict base, classifier delta provides whatever it can
- Integration: effective config's `bm25_weight` is used in RRF fusion (verify same scores as before for identical queries)

**Verification:**
- Existing queries produce identical retrieval results (no behavioral regression)
- Effective config is logged per query
- `_classify_query` returns in <1ms

---

- [x] **Unit 3: HyDE Gating Logic**

**Goal:** Add HyDE enable/disable decision to the classifier based on entity count and query characteristics.

**Requirements:** R1-HG, R2-HG, R3-HG, R6-HG

**Dependencies:** Unit 2 (unified routing profile must exist for `hyde_enabled` field)

**Files:**
- Modify: `src/api.py`

**Approach:**
- In `_classify_query()`, add logic to set `hyde_enabled`:
  - `False` when: entity_hits >= 2 AND no comparison signal (entity-heavy factual, not comparative), or query_tokens < 8 AND entity_hits >= 1 (short precise lookup)
  - `True` (explicit) when: `_DEFINITIONAL_PATTERNS` match and entity_hits == 0 (conceptual), or mode is "outline"/"gaps" (synthesis modes), or `_COMPARE_PATTERNS` match (comparisons benefit from HyDE's bridging vocabulary even with entities)
  - `None` (defer to global config) otherwise
- The gating decision is logged with reason string: `"HyDE disabled: 3 KB entities, no comparison"`, `"HyDE enabled: comparison query"`, `"HyDE: deferred to config"`
- In `_run_retrieval`, the existing `hyde_enabled = _retrieval_cfg.get("hyde_enabled", True)` is replaced by reading from `effective` config (already wired in Unit 2)

**Patterns to follow:**
- Entity counting logic already in `_classify_query_bm25_weight` at line 261
- HyDE gating point at line 873

**Test scenarios:**
- Happy path: query "what mAP does RENet achieve on DSEC?" (2 entities, factual) → `hyde_enabled: False`
- Happy path: query "what is a spiking neural network?" (0 entities, definitional) → `hyde_enabled: True`
- Happy path: query "how do event cameras work?" (0 entities, conceptual) → `hyde_enabled: None` (defers to global config, which is True)
- Edge case: query "compare SpikingJelly to Norse" (2 entities + comparison) → `hyde_enabled: True` (comparison signal overrides entity-heavy gating)
- Edge case: query "SpikingJelly training procedure on DSEC" (2 entities, no comparison) → `hyde_enabled: False` (entity-heavy factual)
- Edge case: query "LIF" (1 entity, 1 token, very short) → `hyde_enabled: False` (short + entity)
- Edge case: query "explain the membrane potential dynamics in biological neurons" (0 entities, long, conceptual) → `hyde_enabled: None` (no strong signal either way)
- Edge case: query "how does deep learning handle event streams" → verify entity_hits count. If common words like "Deep", "Event", "Learning" are in `_kb_entities`, this could false-positive as entity-heavy. The existing stopword filter and CamelCase/acronym regex should exclude these, but verify against actual entity set contents.
- Integration: when `hyde_enabled: False`, verify no HyDE LLM call is made (check logs for absence of "HyDE passage" debug line)

**Verification:**
- Entity-heavy queries skip HyDE (visible in logs, ~500ms latency reduction)
- Conceptual queries still get HyDE
- No regression in retrieval quality for queries where HyDE was previously running and should continue

---

- [x] **Unit 4: Mode-Aware HyDE Prompting**

**Goal:** When HyDE is enabled, condition the generation prompt on the active mode's preferred_sections.

**Requirements:** R4-HG, R5-HG

**Dependencies:** Unit 2 (effective config provides merged `preferred_sections`), Unit 3 (HyDE gating in place)

**Files:**
- Modify: `src/api.py`

**Approach:**
- Define `_HYDE_PROMPTS` dict mapping section categories to prompt strings:
  - `"background"`: "...write a textbook-style explanation as if from a survey paper's background section..."
  - `"results"`: "...write a findings paragraph with specific metrics and comparisons as if from a results section..."
  - `"limitations"`: "...write about limitations, open challenges, and future work directions..."
  - `"generic"`: current prompt (unchanged)
- Add `_select_hyde_prompt(preferred_sections: set | None) -> str`:
  - If None or empty → `"generic"`
  - Map sections to categories: `{"abstract", "introduction", "background", "related work"}` → `"background"`, `{"results", "experiments", "methods"}` → `"results"`, `{"discussion", "conclusion", "limitations", "future work"}` → `"limitations"`. Unmapped sections count toward no category.
  - Pick category by plurality match. Tie-break: `"generic"`
- In `_generate_hyde` (inside `_run_retrieval`), replace the hardcoded system prompt with `_select_hyde_prompt(effective.get("preferred_sections"))`.

**Patterns to follow:**
- Current HyDE prompt at line 882-887
- MODE_ROUTING preferred_sections structure at lines 48-97

**Test scenarios:**
- Happy path: `background` mode → HyDE prompt is textbook-style
- Happy path: `check` mode → HyDE prompt is findings-style with metrics
- Happy path: `gaps` mode → HyDE prompt is limitations/future-work style
- Happy path: `draft` mode (no preferred_sections) → HyDE prompt is generic (current behavior)
- Happy path: default mode (no mode) → HyDE prompt is generic
- Edge case: `compare` mode has `{"results", "experiments", "discussion", "abstract"}` → majority maps to results category

**Verification:**
- HyDE passages match the style of target nuggets for each mode (inspect log output)
- No change to generic prompt for modes without preferred_sections

## System-Wide Impact

- **Interaction graph:** Changes are contained within `src/api.py`. The `/chat` SSE endpoint and `_run_retrieval` are the only affected entry points. MCP server (`src/mcp_server.py`) uses its own retrieval path and is unaffected. The `/retrieve` endpoint also calls `_run_retrieval` — it will benefit from Unit 2 (routing profile) and Unit 3 (HyDE gating) automatically.
- **Error propagation:** Short-circuit fallback ensures no query fails silently. HyDE gating failure (exception in classifier) should default to `hyde_enabled: None` (defer to global config), preserving current behavior.
- **State lifecycle risks:** None. All changes are stateless per-request. No new caches, no new globals beyond `_HYDE_PROMPTS` (a static dict).
- **API surface parity:** The `/retrieve` endpoint (non-SSE) also calls `_run_retrieval` and will inherit routing profile + HyDE gating. Short-circuit only applies to `/chat` (SSE endpoint) since `/retrieve` is used for raw nugget access.
- **Unchanged invariants:** MODE_ROUTING structure, config.yaml schema, SSE response format (except new `retrieval_mode` field in sources event), frontend behavior, MCP server tools.

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| Short-circuit regex false-positives on number-like text | arXiv pattern `\d{4}\.\d{4,5}` is distinctive; paper_id pattern `\d{4}_\d{4,5}` is rare in natural text. Low risk. |
| HyDE gating threshold too aggressive (disables HyDE when it would help) | Start conservative (entity_hits >= 2). Comparison queries exempt. Log all gating decisions for calibration. Easy to adjust threshold. |
| Entity set false positives (common academic words in `_kb_entities`) | Verify entity set contents before shipping. The existing regex filters (CamelCase, acronyms 3+ chars, stopword list) should exclude common words, but validate with queries like "how does deep learning work". |
| Classifier returning bad delta breaks retrieval | All delta fields default to None. Only non-None fields override. Worst case: a bad bm25_weight, which is already the current risk surface. |
| Mode-aware HyDE prompts produce worse passages than generic | Keep generic as fallback. Compare output quality in logs. Can revert individual templates without changing architecture. |

## Sources & References

- **Origin documents:**
  - [Paper-Reference Short-Circuit](docs/brainstorms/2026-04-07-paper-reference-short-circuit-requirements.md)
  - [Unified Routing Profile](docs/brainstorms/2026-04-07-unified-routing-profile-requirements.md)
  - [HyDE Gating](docs/brainstorms/2026-04-07-hyde-gating-requirements.md)
- **Ideation:** [Query-Type Routing Ideation](docs/ideation/2026-04-07-query-type-routing-ideation.md)
- Related code: `src/api.py`, `src/rerank.py`
