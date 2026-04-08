---
date: 2026-04-07
topic: nugget-pipeline-optimization
focus: optimize nugget generation pipeline throughput and quality
---

# Ideation: Nugget Pipeline Optimization

## Codebase Context

RAG knowledge base pipeline processing 1,564 academic papers into ~153k QA nuggets using Qwen3.5-27B via vLLM on HPC (A100 80GB GPUs). Unified pipeline: extract nuggets from chunks -> quality score -> improve weak -> gap-fill sparse. Separate embed stage (Qwen3-Embedding-8B) and store stage (ChromaDB + SQLite). Key bottlenecks: sequential per-chunk extraction (prior_questions dedup), 3 separate LLM stages per paper, full store rebuild, no structured output for extraction.

## Ranked Ideas

### 1. Structured Output for Extraction
**Description:** Use vLLM's `guided_json` / `response_format` with a JSON schema for nugget extraction, replacing the 43-line `repair_json` recovery function. Quality scoring already uses structured output on vLLM; extraction does not.
**Rationale:** Eliminates JSON parse failures, truncated array recovery, markdown fence stripping, and retry overhead. With 100k+ extraction calls, even a 2-3% failure rate means thousands of wasted retries.
**Downsides:** Structured output can slightly increase generation latency. Variable-length array output is more complex than fixed-structure quality output.
**Confidence:** 90%
**Complexity:** Low
**Status:** Selected for immediate implementation

### 2. Manifest-Aware Extraction Prompt
**Description:** Inject paper metadata (title, authors, year, venue, abstract) from the manifest into the extraction system prompt. Give the LLM context about which paper it's reading.
**Rationale:** The static system prompt produces nuggets saying "the proposed method" instead of actual names, which the quality scorer flags and the improvement stage tries to fix. Fixing at extraction time avoids downstream LLM costs. vLLM prefix caching means only the metadata preamble adds new tokens.
**Downsides:** Slightly increases prompt size (~200 tokens per paper). Requires manifest during extraction.
**Confidence:** 90%
**Complexity:** Low
**Status:** Selected for immediate implementation

### 3. Quality-Weighted Retrieval Scoring
**Description:** Integrate existing 6-dimension quality scores into RRF fusion and reranking at query time. Currently quality scores are stored but only used as a tertiary tiebreaker.
**Rationale:** Pure leverage. Quality data exists for all 153k nuggets but is nearly unused. Every query improves with zero re-extraction. Overlaps with quality/confidence gating from the query-type routing ideation.
**Downsides:** Needs calibration to avoid quality dominating relevance signals.
**Confidence:** 85%
**Complexity:** Low
**Status:** Selected for immediate implementation

### 4. Parallel Chunk Extraction with Post-Hoc Dedup
**Description:** Remove sequential `prior_questions` mechanism, extract all chunks in parallel, run post-hoc dedup via embedding similarity or fuzzy matching.
**Rationale:** Biggest throughput bottleneck. 40-chunk paper = 40 sequential LLM calls. Parallel extraction could process in time of 1-2 calls.
**Downsides:** May produce 10-20% more raw duplicates before dedup.
**Confidence:** 85%
**Complexity:** Medium
**Status:** Unexplored

### 5. Single-Call Extraction + Self-Scoring
**Description:** Merge extraction and quality scoring into one LLM call with self-assessed confidence scores.
**Rationale:** Cuts LLM calls by ~30-40%. Current quality scorer rates without source text (epistemically backwards).
**Downsides:** Self-assessment may be less calibrated than external rating.
**Confidence:** 75%
**Complexity:** Medium
**Status:** Unexplored

### 6. Incremental Store Build
**Description:** Replace full-rebuild store with incremental upserts. Track papers in ChromaDB/SQLite, only add/update changed papers.
**Rationale:** Last remaining full-rebuild bottleneck. Adding 5 papers requires rebuilding for all 1,564.
**Downsides:** ChromaDB upsert semantics and FTS5 index updates need careful handling.
**Confidence:** 80%
**Complexity:** Medium
**Status:** Unexplored

### 7. Two-Model Tiered Pipeline
**Description:** Small model for easy chunks (intros, abstracts), full 27B for dense results/comparisons.
**Rationale:** Could halve GPU-hours. Config already supports different models.
**Downsides:** Requires two vLLM instances or model swapping. Needs chunk difficulty heuristic.
**Confidence:** 65%
**Complexity:** High
**Status:** Unexplored

## Rejection Summary

| # | Idea | Reason Rejected |
|---|------|-----------------|
| 1 | Async HTTP (replace ThreadPoolExecutor) | Bottleneck is LLM inference, not HTTP overhead |
| 2 | Pipeline metrics dashboard | Housekeeping, not throughput optimization |
| 3 | Delete legacy code paths | Trivial cleanup, not an optimization |
| 4 | Batched multi-chunk extraction | Context window too tight; complicates parsing |
| 5 | Resume-on-failure checkpointing | Pipeline already skips completed papers |
| 6 | Speculative quality gate | Corpus is user-curated; risks losing nuggets |
| 7 | Eliminate gap-fill | Loses valuable coverage |
| 8 | Drop improvement stage | Loses valuable borderline nuggets |
| 9 | Chunk-level content-hash caching | Niche workflow, diminishing returns |
| 10 | Whole-paper-at-once extraction | Papers exceed context limits; quality degrades |
| 11 | Assertion-based nuggets | Major format change across all consumers |
| 12 | Embed-then-cluster / skip nuggets | Fundamentally different architecture |
| 13 | Direct chunk embedding | Duplicates #12 |
| 14 | Retrieval-augmented extraction | Chicken-and-egg problem |
| 15 | Adaptive prompt per paper domain | Current tiering handles adequately |
| 16 | Embedding-aware gap detection | Post-hoc analysis, not throughput |
| 17 | Feedback-driven quality calibration | Insufficient feedback volume (personal use) |
| 18 | Cross-paper deduplication | Medium value; existing dedup + quality handles bloat |
| 19 | Quality-gated embedding | Unified pipeline already removes low-quality nuggets |

## Session Log
- 2026-04-07: Initial ideation, 38 candidates from 4 agents, 26 after dedup, 7 survivors. User selected ideas 1, 2, 3 for immediate implementation.
