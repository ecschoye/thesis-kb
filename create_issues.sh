#!/usr/bin/env bash
# Create GitHub issues for all identified thesis-kb weak points
set +u
source ~/.bashrc 2>/dev/null || true
set -eo pipefail

REPO="ecschoye/thesis-kb"
API="https://api.github.com/repos/$REPO"
AUTH="Authorization: token $GITHUB_TOKEN"
ACCEPT="Accept: application/vnd.github+json"

create_label() {
  local name="$1" color="$2" desc="$3"
  curl -sf -X POST "$API/labels" \
    -H "$AUTH" -H "$ACCEPT" \
    -d "{\"name\":\"$name\",\"color\":\"$color\",\"description\":\"$desc\"}" \
    > /dev/null 2>&1 || true
}

create_issue() {
  local title="$1" body="$2" labels="$3"
  local payload
  payload=$(python3 -c "
import json, sys
print(json.dumps({
    'title': sys.argv[1],
    'body': sys.argv[2],
    'labels': sys.argv[3].split(',')
}))
" "$title" "$body" "$labels")

  local resp
  resp=$(curl -sf -X POST "$API/issues" \
    -H "$AUTH" -H "$ACCEPT" \
    -d "$payload")
  local url
  url=$(echo "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin)['html_url'])")
  echo "Created: $url  —  $title"
}

echo "=== Creating labels ==="
create_label "P0-critical" "b60205" "Critical priority"
create_label "P1-high"     "d93f0b" "High priority"
create_label "P2-medium"   "fbca04" "Medium priority"
create_label "P3-low"      "0e8a16" "Low priority"
create_label "bug"         "d73a4a" "Something is not working"
create_label "enhancement" "a2eeef" "New feature or request"
create_label "pipeline"    "5319e7" "Batch pipeline (extract/chunk/nuggets/embed/store)"
create_label "api"         "1d76db" "Query server and retrieval"
create_label "ui"          "c5def5" "Frontend / UX"
create_label "schema"      "006b75" "Database schema and storage"
echo "Labels created."

echo ""
echo "=== Creating issues ==="

# --- P0: Critical ---

create_issue \
  "[pipeline] Augmented nuggets never merged into production KB" \
  "## Problem
Quality scoring has run on 1,548 papers and augmentation on 923, but **none of this feeds back into the production database or embeddings**. The KB serves raw, unfiltered nuggets. ~22% of nuggets are flagged as low quality (score ≤ 2) and users see them anyway.

## Affected Files
- \`src/store/kb.py\` — builds KB from raw nuggets only
- \`src/embed/embedder.py\` — has merge logic but it's not wired into the store stage
- \`corpus/nuggets_quality/\` — quality JSON files (unused)
- \`corpus/nuggets_augmented/\` — augmented JSON files (unused)

## Impact
The entire quality/augmentation pipeline is write-only. Users receive low-quality nuggets with no filtering or improvement applied.

## Suggested Fix
1. Add a merge step before embedding: replace improved nuggets, append gap-filled ones
2. Filter out nuggets with overall quality score ≤ threshold (configurable)
3. Wire this into the \`reprocess_pipeline.slurm\` flow" \
  "P0-critical,bug,pipeline"

create_issue \
  "[schema] Quality scores not stored in SQLite — retrieval can't filter" \
  "## Problem
Quality ratings live only in separate JSON files (\`corpus/nuggets_quality/\`). The SQLite database has no quality columns, so retrieval cannot filter or deprioritize low-quality nuggets. The \`confidence\` column in the nuggets table exists but is always empty.

## Affected Files
- \`src/store/kb.py\` — schema definition, nugget insertion
- \`src/query.py\` — retrieval queries (no quality filtering possible)
- \`src/nuggets/quality.py\` — produces scores but doesn't write to DB

## Impact
Cannot filter by quality at query time. 22% of flagged nuggets served to users.

## Suggested Fix
1. Add columns to nuggets table: \`overall_score\`, \`thesis_relevance_score\`, \`flagged\` (boolean)
2. Populate from quality JSON during store stage
3. Add WHERE clause in \`query.py\` to filter \`overall_score >= threshold\`
4. Populate the existing \`confidence\` column or remove it" \
  "P0-critical,enhancement,schema"

create_issue \
  "[api] Feedback DB concurrent writes can corrupt SQLite" \
  "## Problem
\`feedback.db\` is opened with \`check_same_thread=False\` (api.py ~line 245) allowing concurrent access, but SQLite is not concurrent-write safe. Multiple simultaneous requests writing feedback can corrupt the database.

## Affected Files
- \`src/api.py\` — feedback DB connection and write endpoints

## Impact
Data corruption under concurrent load. Feedback data loss.

## Suggested Fix
- Enable WAL mode: \`PRAGMA journal_mode=WAL\`
- Use a write queue (asyncio.Queue) or a dedicated writer thread
- Or switch to a separate feedback store (e.g., append-only JSON log)" \
  "P0-critical,bug,api"

create_issue \
  "[query] Silent exception swallowing masks retrieval failures" \
  "## Problem
Multiple critical code paths catch all exceptions and return empty results with no logging:
- \`bm25_search()\` (~line 116): \`except Exception: return []\` — if FTS5 breaks, user gets empty results silently
- \`load_chunk()\` (~line 132): silent catch masks file I/O errors
- ChromaDB initialization can fail with no error handling

## Affected Files
- \`src/query.py\` — \`bm25_search()\`, \`load_chunk()\`

## Impact
Retrieval failures are invisible. Users get degraded results with no indication anything went wrong. Impossible to diagnose issues.

## Suggested Fix
1. Add proper logging to all except blocks (use \`src/log.py\`)
2. Distinguish recoverable vs fatal errors
3. Surface retrieval health to the API (e.g., \`/health\` endpoint)" \
  "P0-critical,bug,api"

create_issue \
  "[query] BM25 query not sanitized for FTS5 special characters" \
  "## Problem
Raw user queries are passed directly to SQLite FTS5 \`MATCH\` operator without validation or escaping. FTS5 special characters (\`AND\`, \`OR\`, \`NOT\`, \`\"\`, \`*\`, \`NEAR\`) in user input cause query failures that are silently swallowed.

## Affected Files
- \`src/query.py\` — \`bm25_search()\` method

## Impact
Queries containing common words like \"NOT\" or quoted phrases fail silently, returning empty results.

## Suggested Fix
1. Escape or quote user input before passing to FTS5 MATCH
2. Wrap each term in double quotes: \`\"term1\" \"term2\"\`
3. Log and surface FTS5 errors instead of swallowing them" \
  "P0-critical,bug,api"

# --- P1: High ---

create_issue \
  "[api] All retrieval tuning parameters hardcoded in Python" \
  "## Problem
Critical retrieval parameters are scattered as magic numbers throughout \`src/api.py\`:
- RRF constant \`+30\` (lines 734, 756) — undocumented, not in config
- \`MODE_ROUTING\` dict with per-mode \`n_retrieve\`, \`authority_boost\`, \`max_per_paper\` — requires code change + restart
- Boosting coefficients: \`1.15\` (section), \`1.3/0.5/0.05\` (feedback), \`log2/5\` (depth)
- Reranking \`blend_weight=0.6\`, \`top_n=60\`

## Affected Files
- \`src/api.py\` — MODE_ROUTING, boost functions, rerank call
- \`config.yaml\` — missing these parameters

## Impact
Cannot tune retrieval without code changes and server restart. No systematic optimization possible.

## Suggested Fix
Move all parameters to \`config.yaml\` under a \`retrieval:\` section with per-mode overrides." \
  "P1-high,enhancement,api"

create_issue \
  "[extract] OCR fallback not implemented despite config flag" \
  "## Problem
\`config.yaml\` has \`ocr_fallback: true\` but the PDF extraction code simply skips scanned pages, returning empty strings. Papers with mixed text/scanned content lose pages silently.

## Affected Files
- \`src/extract/pdf_to_text.py\` — page extraction logic
- \`config.yaml\` — \`ocr_fallback\` flag (non-functional)

## Impact
Papers with scanned figures, tables, or pages have missing content. Unknown how many papers are affected.

## Suggested Fix
1. Implement OCR fallback using \`pytesseract\` or \`easyocr\` for pages with <100 chars extracted
2. Or remove the misleading config flag and document the limitation
3. Add metrics: count and log pages skipped per paper" \
  "P1-high,enhancement,pipeline"

create_issue \
  "[chunk] Token-only splitting breaks mid-sentence/table/equation" \
  "## Problem
Chunking splits on token count alone (400 tokens, 100 overlap) with no semantic boundary detection. This breaks:
- Mid-sentence splits
- Tables split across chunks
- Equations and code blocks broken
- Lists cut in half

## Affected Files
- \`src/chunk/chunker.py\` — chunking logic

## Impact
Malformed chunk boundaries produce poor context for LLM nugget extraction. Nuggets from split content are often incoherent.

## Suggested Fix
1. Add sentence boundary detection (split on \`.\` / \`\\n\\n\` near token boundary)
2. Detect table/equation blocks and keep them intact
3. Allow chunk size to flex ±10% to hit natural boundaries" \
  "P1-high,enhancement,pipeline"

create_issue \
  "[extract] Multi-column PDF text ordering interleaved" \
  "## Problem
PyMuPDF uses physical layout order for text extraction. Two-column papers (common in IEEE, ACL, CVPR formats) get left and right column text interleaved, breaking sentence and paragraph continuity.

## Affected Files
- \`src/extract/pdf_to_text.py\` — \`get_text()\` calls

## Impact
Interleaved column text produces nonsensical chunks and nuggets for a significant portion of the corpus (most conference papers are two-column).

## Suggested Fix
1. Use \`sort=True\` flag in PyMuPDF \`get_text()\` for reading-order extraction
2. Or use \`pymupdf4llm\` which handles column detection
3. Or detect multi-column layout and process columns separately" \
  "P1-high,bug,pipeline"

create_issue \
  "[api] No query or retrieval logging/instrumentation" \
  "## Problem
The query pipeline has zero logging or instrumentation. Cannot see:
- What queries users run
- Which modes are used most
- Retrieval latency per stage (expansion, embedding, vector search, BM25, reranking)
- Which retrieval parameters produce best results
- Error rates

## Affected Files
- \`src/api.py\` — all retrieval endpoints
- \`src/query.py\` — ThesisKB class (no logging at all)
- \`src/rerank.py\` — no timing or metrics

## Impact
Cannot diagnose issues, measure quality, or optimize retrieval. Flying blind.

## Suggested Fix
1. Add structured logging with \`src/log.py\` at each pipeline stage
2. Log: query text, mode, expansion variants, result counts, latencies, rerank scores
3. Optional: persist query logs to SQLite for analysis" \
  "P1-high,enhancement,api"

create_issue \
  "[api] Type balancing allows max_per_paper overrun" \
  "## Problem
In the nugget selection logic (api.py ~lines 914-939), after type-balanced selection (max 2 per type), the backfill loop adds remaining nuggets without respecting the \`max_per_paper\` limit. A single paper can dominate the context window.

## Affected Files
- \`src/api.py\` — nugget selection / type balancing logic

## Impact
One paper with many high-scoring nuggets can crowd out diversity, giving the LLM a skewed view of the literature.

## Suggested Fix
Enforce \`max_per_paper\` in the backfill loop. Track per-paper counts across both the type-balanced and backfill phases." \
  "P1-high,bug,api"

# --- P2: Medium ---

create_issue \
  "[schema] Missing database indexes on thesis_relevance, section, type" \
  "## Problem
The SQLite nuggets table lacks indexes on frequently-queried columns:
- \`thesis_relevance\` — used for filtering
- \`section\` — used for section-biased retrieval
- No composite index on \`(paper_id, type)\`

Existing indexes: \`idx_nuggets_paper\`, \`idx_nuggets_type\` only.

## Affected Files
- \`src/store/kb.py\` — schema creation

## Impact
Slower queries as KB grows, especially for filtered retrieval.

## Suggested Fix
Add indexes:
\`\`\`sql
CREATE INDEX idx_nuggets_relevance ON nuggets(thesis_relevance);
CREATE INDEX idx_nuggets_section ON nuggets(section);
CREATE INDEX idx_nuggets_paper_type ON nuggets(paper_id, type);
\`\`\`" \
  "P2-medium,enhancement,schema"

create_issue \
  "[embed] No metadata in nugget embeddings" \
  "## Problem
Nugget embeddings encode only \`\"Q: {question} A: {answer}\"\`. Paper ID, section, nugget type, and page numbers are not part of the embedded text. This means vector search cannot leverage metadata signals.

## Affected Files
- \`src/embed/embedder.py\` — embedding text construction

## Impact
Vector search is purely content-based. Cannot do type-biased or section-biased retrieval at the embedding level — must rely entirely on post-hoc filtering.

## Suggested Fix
Include nugget type and section in the embedded text:
\`\"[{type}] [{section}] Q: {question} A: {answer}\"\`
Requires re-embedding the entire KB." \
  "P2-medium,enhancement,pipeline"

create_issue \
  "[ui] No retrieval parameter controls in frontend" \
  "## Problem
Users cannot adjust any retrieval parameters from the UI:
- \`n_retrieve\`, \`n_context\`
- Reranking on/off, weight, top_n
- \`max_per_paper\`
- Year range filters (beyond basic controls)

All locked to backend defaults per mode.

## Affected Files
- \`static/index.html\` — React frontend
- \`src/api.py\` — needs to accept parameter overrides

## Impact
Power users can't tune retrieval for their specific query needs. No way to experiment with settings.

## Suggested Fix
Add an \"Advanced\" collapsible panel with sliders/inputs for key parameters. Pass as query params to the API." \
  "P2-medium,enhancement,ui"

create_issue \
  "[ui] Query expansion variants not visible to user" \
  "## Problem
The LLM generates ~6 search variant queries per user query, but these are never shown to the user. Users don't know what was actually searched or why certain results appeared.

## Affected Files
- \`src/api.py\` — expansion logic (already generates variants)
- \`static/index.html\` — no UI for displaying variants

## Impact
Low transparency. Users can't understand or debug why retrieval returned specific results.

## Suggested Fix
Return expansion variants in the SSE stream metadata. Display them in a collapsible section above results." \
  "P2-medium,enhancement,ui"

create_issue \
  "[ui] No chunk context highlighting for nugget source" \
  "## Problem
When viewing source chunks, the nugget's position within the chunk text is not highlighted. Users see a large block of text without knowing where the nugget was extracted from.

## Affected Files
- \`static/index.html\` — chunk display component

## Impact
Hard to verify nugget accuracy against source text. Users must manually scan the chunk.

## Suggested Fix
Fuzzy-match the nugget answer against the chunk text and highlight the matching span." \
  "P2-medium,enhancement,ui"

create_issue \
  "[api] ThreadPoolExecutor undersized at 4 workers" \
  "## Problem
\`ThreadPoolExecutor(max_workers=4)\` is hardcoded in \`api.py\` (line 34). This single executor is shared between embedding, retrieval, reranking, and LLM calls. Under concurrent requests, tasks queue up causing head-of-line blocking.

## Affected Files
- \`src/api.py\` — executor initialization

## Impact
Slow response times under concurrent load. No visibility into queue depth.

## Suggested Fix
1. Make \`max_workers\` configurable via \`config.yaml\`
2. Consider separate executors for CPU-bound (reranking) vs I/O-bound (LLM calls) work
3. Add queue depth monitoring" \
  "P2-medium,enhancement,api"

create_issue \
  "[embed] No embedding cache for repeated queries" \
  "## Problem
Query expansion generates ~6 variants per query, each embedded separately. The same variant queried multiple times in a session is re-embedded every time. No caching layer exists.

## Affected Files
- \`src/api.py\` — query embedding calls
- \`src/embed/embedder.py\` — no cache

## Impact
Unnecessary latency and compute cost for repeated or similar queries.

## Suggested Fix
Add an LRU cache on the embedding function keyed by input text. Even a small cache (256 entries) would help for session-level deduplication." \
  "P2-medium,enhancement,api"

create_issue \
  "[api] Cross-encoder reranker has no timeout or fallback" \
  "## Problem
Cross-encoder inference on up to 60 candidates can hang if the model is overloaded. There is no configurable timeout and no fallback to RRF-only ranking if reranking fails. Model loading failures at \`Ranker(model_name=...)\` are also unhandled.

## Affected Files
- \`src/rerank.py\` — \`rerank_nuggets()\`
- \`src/api.py\` — rerank call site

## Impact
A hung reranker stalls the entire response indefinitely. A failed model load crashes the endpoint.

## Suggested Fix
1. Wrap reranking in a timeout (configurable, e.g., 10s)
2. Fallback to RRF-only ranking on timeout or error
3. Try-catch model loading with clear error message" \
  "P2-medium,enhancement,api"

create_issue \
  "[schema] Paper metadata duplicated across 4+ sources" \
  "## Problem
Paper metadata (title, authors, year) is stored redundantly in:
1. \`corpus/manifest.json\`
2. SQLite \`papers\` table
3. Nugget JSON files (\`paper_title\`, \`paper_authors\`, \`paper_year\` fields)
4. ChromaDB metadata
5. Embeddings JSON (\`nuggets_with_embeddings.json\`)

No single source of truth. Easy to get out of sync.

## Affected Files
- \`src/store/kb.py\` — writes to SQLite + ChromaDB
- \`src/nuggets/extract.py\` — writes to nugget JSON
- \`src/embed/embedder.py\` — writes to embeddings JSON

## Impact
Metadata inconsistency across sources. Updates to paper info must be applied in 4+ places.

## Suggested Fix
Designate SQLite \`papers\` table as the single source of truth. Other stores reference \`paper_id\` only and join at query time." \
  "P2-medium,enhancement,schema"

echo ""
echo "=== Done! All issues created. ==="
