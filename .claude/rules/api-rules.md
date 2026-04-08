---
paths:
  - "src/api.py"
---

# API Rules

- All responses use SSE (Server-Sent Events) streaming — never return plain JSON for query endpoints
- Mode-based retrieval: each slash command (background, draft, check, review, compare, gaps, outline) configures different retrieval parameters (n_retrieve, preferred_sections, authority_boost, max_per_paper)
- Keep the API stateless — no session storage, no in-memory caches that grow unbounded
- Use FastAPI's dependency injection for KB client, embedding client, and config
- Error responses must also be SSE events (type: "error") so the frontend can handle them uniformly
- Cross-encoder reranking (bge-reranker-v2-m3 via transformers) runs after RRF fusion — do not skip this step
- Query classification (`_classify_query`) returns a routing delta dict that layers on top of MODE_ROUTING via `_build_effective_config` — all retrieval knobs (bm25_weight, hyde_enabled, blend_weight, n_retrieve, etc.) are driven by this effective config
- Paper-reference short-circuit in the /chat handler skips full retrieval for single-paper queries (cite keys, arXiv IDs, paper_ids)
