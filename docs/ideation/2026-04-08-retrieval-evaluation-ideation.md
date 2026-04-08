---
date: 2026-04-08
topic: retrieval-evaluation
focus: ground truth and metrics for measuring retrieval quality
---

# Ideation: Retrieval Evaluation Infrastructure

## Codebase Context

RAG knowledge base for MSc thesis. 162k nuggets from 1,735 papers. Zero ground truth query-nugget relevance pairs. Every retrieval optimization (query-type routing, HyDE gating, quality-weighted scoring, reranker blend weights) was tuned by intuition. At least 5 parameters in docs are explicitly marked "needs empirical calibration" with no way to calibrate them. feedback.db exists with sparse thumbs up/down (single user). Quality scores measure nugget quality, not query-relevance. Cross-encoder scores computed but discarded at inference time.

## Ranked Ideas

### 1. Priority Topic Canary Queries
**Description:** 5 hand-curated queries per retrieval mode (35 total) with known expected papers/nugget types. JSON fixture. Run after store rebuild or config change.
**Rationale:** Cheapest regression guard. One afternoon of curation protects all 7 modes against silent breakage.
**Downsides:** Only catches gross failures, not subtle degradation.
**Confidence:** 90%
**Complexity:** Low
**Status:** Selected for immediate brainstorm

### 2. Retrieval Trace Logger
**Description:** Structured JSONL log per query: expanded variants, per-source scores, boost factors, cross-encoder scores, final ranking.
**Rationale:** Observability foundation. Cross-encoder scores are computed but discarded. Every downstream improvement depends on this.
**Downsides:** Log volume, minor I/O on hot path.
**Confidence:** 90%
**Complexity:** Low
**Status:** Selected (after canary queries)

### 3. Nugget Self-Retrieval Test
**Description:** Use nugget question fields as queries, check whether source nugget appears in top-k. 162k implicit pairs, zero labeling.
**Rationale:** Fastest path to baseline MRR/recall@k. Exposes embedding blind spots and broken nuggets.
**Downsides:** Tests self-consistency, not real user queries.
**Confidence:** 90%
**Complexity:** Low
**Status:** Selected (after trace logger)

### 4. Thesis-Section Grounded Eval Set
**Description:** Parse thesis paragraphs with \cite{} keys, use claim text as queries, check cited papers in top-k.
**Rationale:** Highest ecological validity. Real information needs already validated by the thesis author.
**Downsides:** Requires written thesis paragraphs with citations.
**Confidence:** 85%
**Complexity:** Medium
**Status:** Selected (after self-retrieval test)

### 5. LLM-as-Judge Relevance Annotation
**Description:** Offline script rating (query, nugget) pairs via LLM. Bootstrap ground truth from retrieval traces.
**Rationale:** Scalable ground truth. quality.py provides template for structured LLM evaluation.
**Downsides:** LLM judge biases. Requires trace logger.
**Confidence:** 75%
**Complexity:** Medium
**Status:** Deferred (depends on trace logger + eval set)

### 6. Parameter Sweep Framework
**Description:** Replay eval queries with parameter grid, report NDCG@10/recall@10 per config.
**Rationale:** Directly addresses "5+ parameters need calibration." Compounds with every new parameter.
**Downsides:** Requires eval set.
**Confidence:** 80%
**Complexity:** Medium
**Status:** Deferred (depends on eval set)

### 7. Mode-Stratified Evaluation
**Description:** Mode-specific success metrics: background=diversity, check=precision, gaps=limitation nuggets.
**Rationale:** Global metrics mask mode-specific regressions.
**Downsides:** Multiplies eval set size needed.
**Confidence:** 75%
**Complexity:** Medium
**Status:** Deferred (depends on eval set + sweep)

## Rejection Summary

| # | Idea | Reason Rejected |
|---|------|-----------------|
| 1 | Overlap-count as precision signal | Weak proxy; agreement ≠ relevance |
| 2 | Citation-back-reference | LLM citation depends on prompt/style, fragile |
| 3 | Feedback mining + amplification | Insufficient volume (single user) |
| 4 | Query expansion ablation | Downstream of eval set |
| 5 | Nugget cluster coherence | Tests embeddings, not retrieval |
| 6 | Abstract-as-query retrieval | Trivial case, doesn't match real queries |
| 7 | Mode-differential consistency | No ground truth, just consistency |
| 8 | Nugget consensus health check | Only works for factual queries |
| 9 | Citation coverage gap | Measures KB coverage, not retrieval; already exists |
| 10 | Embedding drift anchor pairs | Tests embeddings, not retrieval |
| 11 | "Would I cite this?" metric | Not automatable |
| 12 | Quality score vs relevance validation | Depends on eval set |
| 13 | Retrieval regression CI gate | Premature; depends on eval set + sweep |
| 14 | Boost factor profiler | Depends on trace logger |
| 15 | Contrastive paper pair probes | Complex to construct cleanly |
| 16 | Feedback contrastive pairs | Same volume problem |

## Implementation Order

```
Canary Queries (immediate, independent)

Trace Logger → Self-Retrieval Test → Thesis-Grounded Eval Set
                                          ↓
                                    LLM-Judge Annotations
                                          ↓
                                    Parameter Sweep
                                          ↓
                                  Mode-Stratified Eval
```

## Session Log
- 2026-04-08: Initial ideation, 40 candidates from 4 agents, 23 after dedup, 7 survivors. User selected ideas 1-4 for sequential implementation (5→3→1→2), deferred 5-7.
