---
date: 2026-04-07
topic: query-type-routing
focus: "RRF weights are static across all query types. Definitional vs comparative vs factual queries likely benefit from different retrieval balances."
---

# Ideation: Query-Type Routing

## Codebase Context

RAG system for MSc thesis: 1,564 papers, ~153k QA nuggets. Query pipeline: expand -> embed -> retrieve (vector + BM25) -> RRF fusion -> cross-encoder rerank -> LLM stream.

Current query-adaptive state:
- `_classify_query_bm25_weight()` adjusts BM25 weight (0.5-2.0x) via regex patterns + KB-derived entity count (~5,400 terms)
- Mode-based configs per slash command set n_retrieve, preferred_sections, authority_boost, max_per_paper
- HyDE generates hypothetical answers, gated by static boolean
- Cross-encoder reranker (bge-reranker-v2-m3) uses fixed blend_weight
- Rich nugget metadata (type, section, confidence, thesis_relevance, overall_score) largely unused in retrieval

Key gap: Only BM25 weight adapts per-query. 7 other tunable knobs (hyde_enabled, hyde_weight, blend_weight, n_retrieve, section_prefs, authority_boost, max_per_paper) are static within a mode.

## Ranked Ideas

### 1. Paper-Reference Short-Circuit
**Description:** When a query references a specific paper (cite key, arXiv ID, or recognized title), skip expand/embed/retrieve and go directly to `get_paper_nuggets` filtered by mode's `allowed_types`. Infrastructure exists: `_extract_cite_keys`, `_resolve_cite_to_paper_id` in api.py, cite_prefetch hook. The pipeline just never short-circuits.
**Rationale:** Turns multi-second retrieval into millisecond DB lookup for the most precise query type. These queries cannot benefit from vector search or HyDE.
**Downsides:** Must detect paper references reliably. Edge case: user mentions a paper but wants cross-paper context ("how does X compare to others").
**Confidence:** 75%
**Complexity:** Low
**Status:** Explored (2026-04-07)

### 2. Query-Adaptive HyDE Gating
**Description:** Disable HyDE for entity-heavy factual queries (2+ KB entities, short length) and definitional lookups. When HyDE runs, condition the prompt on target section type (background -> textbook-style, results -> findings-style). The classifier already identifies these query classes; HyDE is gated only by a static boolean.
**Rationale:** HyDE costs ~500ms LLM round-trip and hallucinated details for factual queries pull in irrelevant results. "What mAP does RENet achieve on DSEC?" produces a hypothetical answer with invented numbers. Gating + section-aware prompting improves precision and cuts latency.
**Downsides:** Must define gating threshold carefully. Some entity-rich queries still benefit from HyDE (conceptual comparisons). False negatives harder to detect than false positives.
**Confidence:** 80%
**Complexity:** Low
**Status:** Explored (2026-04-07)

### 3. Unified Query Routing Profile
**Description:** Replace `_classify_query_bm25_weight()` (returns float) with a function returning a full routing dict: `{bm25_weight, hyde_enabled, hyde_weight, blend_weight, n_retrieve, section_prefs, authority_boost, max_per_paper}`. Same entity count + regex signals drive all knobs. MODE_ROUTING already consumes these parameters.
**Rationale:** Architectural enabler for ideas 4 and 5. One classifier -> one dict -> every downstream stage adapts. Currently only 1 of 8 knobs responds to query characteristics.
**Downsides:** Requires careful defaults so bad classification doesn't degrade results. More parameters to tune. Risk of coupling too many decisions to a single classifier.
**Confidence:** 85%
**Complexity:** Medium
**Status:** Explored (2026-04-07)

### 4. Quality/Confidence Score Gating
**Description:** Use existing `overall_score` and `confidence` fields (stored in SQLite/ChromaDB but never used in retrieval) as scoring signals. High-stakes modes (check, review) prefer high-quality nuggets; exploratory modes (gaps, outline) tolerate lower quality. Add `quality_floor` to routing config and confidence-based boost factor.
**Rationale:** 153k nuggets vary enormously in quality. The unified pipeline spent LLM compute producing quality scores that are completely wasted in retrieval. Most concrete waste in the pipeline.
**Downsides:** Quality scores may have their own biases. Setting floor too high excludes valuable long-tail nuggets. Needs calibration against actual score distribution.
**Confidence:** 70%
**Complexity:** Low
**Status:** Unexplored

### 5. Adaptive Reranker Blend Weight
**Description:** Make cross-encoder `blend_weight` query-adaptive. High `overlap_count` (RRF and BM25 agree) -> lower blend to trust initial ranking. Low overlap -> higher blend to let cross-encoder resolve conflict. `overlap_count` is already computed (line 981) but never used in scoring.
**Rationale:** Same pattern as idea 4: infrastructure exists, signal is computed, but never used. Reranker excels at semantic relevance but can hurt keyword-heavy queries.
**Downsides:** Heuristic on top of heuristic. Overlap-to-optimal-blend relationship isn't guaranteed monotonic. Needs empirical calibration.
**Confidence:** 65%
**Complexity:** Low
**Status:** Unexplored

### 6. Section as First-Class Routing Signal
**Description:** Promote `preferred_sections` from 1.15x post-retrieval boost to two-tier: ChromaDB `where` filter first, soft boost fallback if too few results. Also infer section preferences from query content in default (no-mode) chat via keyword detection.
**Rationale:** 15% boost too weak to overcome high vector similarity from wrong section. ChromaDB `where` clauses are faster than post-filtering. Default chat gets no section filtering at all.
**Downsides:** Section labels from extraction aren't always accurate. Hard filtering can be too aggressive. Fallback logic adds branching. Medium complexity for uncertain signal quality.
**Confidence:** 75%
**Complexity:** Medium
**Status:** Unexplored

### 7. Conversation-Aware Retrieval Context
**Description:** For multi-turn chats, carry forward entity context from prior turns. Apply entity set to full chat history to maintain a "topic stack" biasing retrieval. Suppress nuggets already surfaced in previous turns.
**Rationale:** Chat endpoint receives full history but only uses `last_msg` for retrieval. Thesis-writing sessions are inherently multi-turn.
**Downsides:** Topic drift detection is hard. Suppressing previously-surfaced nuggets could hurt on intentional revisits. Adds state tracking to stateless API. Medium-high complexity for a single-user tool.
**Confidence:** 60%
**Complexity:** Medium-High
**Status:** Unexplored

## Rejection Summary

| # | Idea | Reason Rejected |
|---|------|-----------------|
| 1 | LLM-based query intent classifier | User explicitly rejected LLM classifier; too expensive for marginal gain |
| 2 | Dynamic n_retrieve by specificity | Subsumed by unified routing profile (idea 3) |
| 3 | Query-variant dedup by embedding similarity | Marginal gain; expansion rarely produces true duplicates |
| 4 | Negative retrieval for contrastive queries | Complex to implement correctly; narrow benefit |
| 5 | Retrieval confidence feedback loop | Two-pass retrieval adds significant latency |
| 6 | Multi-stage retrieval with narrowing | Architecturally complex for single-user system |
| 7 | Entity-aware query decomposition | Overlaps with unified routing + section routing |
| 8 | Feedback-driven routing adaptation | Insufficient query volume for a single-user thesis tool |
| 9 | Citation graph expansion | Requires citation data not in manifest; separate feature |
| 10 | Per-variant expansion confidence weighting | Requires prompt changes; marginal signal |
| 11 | Late-fusion mode selection | Loses efficiency of early filtering |
| 12 | Nugget-type distribution as retrieval signal | Post-hoc correction is fragile; better to route correctly up front |
| 13 | Route on answer diversity not query type | Interesting reframe but not actionable without concrete mechanism |
| 14 | Adaptive RRF-k based on agreement | k-tuning is subtle; overlap_count is noisy for this purpose |
| 15 | Query decomposition for compound questions | Overlaps with expansion; narrow query class |
| 16 | Temporal-aware routing | Useful but narrow; foldable into unified routing dict |
| 17 | Entropy-based retrieval termination | ChromaDB doesn't support early termination natively |
| 18 | Overlap-aware diversity penalty | Already partially handled by max_per_paper |

## Session Log
- 2026-04-07: Initial ideation -- 39 candidates generated (4 agents), 25 after dedup, 7 survived. User re-ranked by assessed impact for thesis-kb: paper-reference short-circuit first (easiest win), HyDE gating second (addresses prior calibration concern), unified routing third (enabler), quality gating and adaptive blend next (unused infrastructure), section routing and conversation context last (complexity vs. signal concerns).
- 2026-04-07: Brainstorming ideas 7, 2, 1 sequentially.
