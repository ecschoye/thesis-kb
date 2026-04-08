---
title: "refactor: Switch nugget output format from JSON to JSONL"
type: refactor
status: active
date: 2026-04-07
deepened: 2026-04-07
---

# refactor: Switch nugget output format from JSON to JSONL

## Overview

Switch per-paper nugget output files and the aggregated KB file from JSON (single object/array) to JSONL (one JSON object per line). This enables streaming reads, reduces peak memory for the 150k+ nugget aggregated file, and simplifies append-based writing during extraction.

## Problem Frame

The current pipeline writes each paper's nuggets as a single JSON object with an envelope:

```json
{"paper_id": "...", "num_nuggets": N, "nuggets": [...], "removed": [...]}
```

The aggregated file `kb/nuggets_with_embeddings.json` is a single JSON array of all nuggets (~150k objects). Both formats require loading the entire file into memory to read any part of it. JSONL eliminates this by making each line independently parseable.

## Requirements Trace

- R1. Per-paper nugget files (`.jsonl`) write one nugget object per line
- R2. The aggregated KB file (`nuggets_with_embeddings.jsonl`) writes one nugget per line
- R3. All active downstream consumers (embed, store, unified reprocess/review) read the new format
- R4. Skip-check logic works without parsing full file content
- R5. Metadata previously in the envelope (`num_nuggets`, `quality_summary`, `removed`) is preserved or derivable

## Scope Boundaries

- No backwards-compatibility shim for reading old JSON files. Reprocessing is the intended recovery path (delete output, rerun stage).
- `src/code_extract/nuggets.py` is out of scope — it writes a bare JSON list but has no downstream consumer in the main pipeline. Migrating it is a consistency-only change that adds review noise.
- **Legacy stages (`quality.py`, `augment.py`, `enrich.py`) are out of scope.** Per CLAUDE.md, quality and augment "EXISTS but NEVER RUN" — they are superseded by the unified pipeline. Enrich operates on the same envelope keys (`improved`, `gap_filled`) from augment output. These files will be updated only if/when the legacy pipeline is revived.
- MCP server and tools are unaffected (they read from SQLite/ChromaDB, not raw files).
- The `config.yaml` paths remain extension-agnostic (they point to directories, not files).
- `store/kb.py`'s quality score reader (`corpus/nuggets_quality/`) stays JSON — those files are produced by the legacy quality stage which is out of scope.

## Context & Research

### Relevant Code and Patterns

- `src/utils.py:save_json` / `load_json` — current atomic-write JSON utilities (includes `os.makedirs` for parent dirs)
- `src/embed/embedder.py:load_all_nuggets()` — main aggregator, reads `data.get("nuggets", [])` from per-paper files
- `src/nuggets/extract.py` — writes envelope to `corpus/nuggets/{paper_id}.json`
- `src/nuggets/unified.py` — writes envelope to `corpus/nuggets_unified/{paper_id}.json`
- `src/nuggets/unified.py:_select_oldest()` — scans unified_dir for `.json` files (used by `--oldest` flag)
- `src/store/kb.py` — reads `kb/nuggets_with_embeddings.json` as flat list
- Skip-check functions: `_nugget_done()` in extract.py, `_done()` in unified.py

### Key Observations

1. Every per-paper consumer accesses `data.get("nuggets", [])` — the envelope is just a wrapper
2. `paper_id` is already present on each nugget object — no data loss dropping the envelope
3. `removed` nuggets are write-only (never read by downstream stages) — audit trail only
4. `quality_summary` and `num_nuggets` are derivable from the nugget list itself
5. `kb/nuggets_with_embeddings.json` is the largest file (150k+ objects) — biggest memory win
6. unified.py has multiple read paths: `_on_result` (write), `_done` (skip check), `_select_oldest` (file scan), and reprocess/review input reading (~line 604-606)

## Key Technical Decisions

- **JSONL for both per-paper and aggregated files**: Consistent format across the pipeline. Per-paper files are small (50-300 lines), but uniformity reduces cognitive overhead.
- **Drop the envelope wrapper**: Each nugget already carries `paper_id`. `num_nuggets` is derivable from line count. `quality_summary` can be computed on the fly.
- **Removed nuggets as flagged lines**: Nuggets with `"_removed": true` are written as JSONL lines at the end of the file, preserving the audit trail without a separate structure. Only the embedder needs to filter these; store reads the already-filtered aggregated file.
- **File extension `.jsonl`**: Per-paper files become `{paper_id}.jsonl`. The aggregated file becomes `nuggets_with_embeddings.jsonl`.
- **Skip-check semantic change (intentional)**: Current behavior reprocesses papers with 0 nuggets on every run (writes `num_nuggets: 0`, skip check requires `> 0`). New behavior: an empty `.jsonl` file is a valid "done" marker — if a paper had no extractable content, reprocessing won't help. To reprocess, delete the file.
- **No dual-format reader**: Clean cut. Old `.json` files are ignored; rerun the stage to regenerate.
- **Legacy stages out of scope**: quality.py, augment.py, and enrich.py are never run in the active pipeline. Their skip checks in extract.py and unified.py still need updating (they check unified_dir), but the legacy modules themselves are untouched.
- **Incremental embed on first post-migration run**: `_load_existing_kb()` checks for `nuggets_with_embeddings.jsonl` + `embeddings.npy`. On first run after migration, the `.jsonl` file won't exist (only the old `.json`). `_load_existing_kb` returns `(None, None)`, and `run_embedding` sets `incremental = False` (line 303), causing a full re-embed of all nuggets. This is a one-time cost, not a silent data loss.

## Open Questions

### Resolved During Planning

- **What happens to `quality_summary`?** It is only consumed by logging in unified.py. The individual nugget `quality` dicts (already per-nugget) are what store.py actually uses. Drop `quality_summary` from the file; compute it in-memory if needed for logging.
- **What about `num_improved` / `num_gap_filled` envelope fields?** These are only used for progress logging during unified processing. Compute from the nugget list in-memory after processing, before printing stats.
- **Should `save_jsonl` do atomic writes?** Yes, same pattern as `save_json` (write to `.tmp`, then `os.replace`, with `os.makedirs` for parent dirs).
- **Should quality.py/augment.py/enrich.py migrate?** No. They are legacy stages superseded by the unified pipeline. Migrating dead code adds risk with no benefit.
- **Do consumers need to filter `_removed`?** Only the embedder (`load_all_nuggets`). The store reads the already-filtered aggregated file. The unified reprocess/review mode reads its own output, which may contain `_removed` lines — but it already handles removed nuggets in its processing logic.

### Deferred to Implementation

- **Exact line-counting vs size-check for skip logic**: Implementation will determine whether line count or simple `os.path.exists()` is sufficient.

## Implementation Units

- [ ] **Unit 1: Add JSONL utilities to src/utils.py**

**Goal:** Provide `save_jsonl()` and `load_jsonl()` functions alongside existing JSON utilities.

**Requirements:** R1, R2, R3

**Dependencies:** None

**Files:**
- Modify: `src/utils.py`

**Approach:**
- `save_jsonl(items, path, removed=None)`: Write each item as one JSON line (no indent, `ensure_ascii=False`). If `removed` is provided, append those with `"_removed": true`. Use atomic write (`.tmp` + `os.replace`). Include `os.makedirs` for parent directory, matching `save_json`.
- `load_jsonl(path)`: Read file line by line, `json.loads()` each non-empty line, return list. Skip blank lines.
- `count_jsonl(path)`: Return non-empty line count without parsing (for skip checks and logging).

**Patterns to follow:**
- `save_json` / `load_json` in same file — same atomic write pattern with `os.makedirs`

**Test scenarios:**
- Happy path: save 3 nuggets + 1 removed, load them back, verify count and `_removed` flag
- Edge case: empty list produces empty file, `load_jsonl` returns `[]`
- Edge case: file with blank lines — `load_jsonl` skips them

**Verification:**
- `save_jsonl` followed by `load_jsonl` round-trips correctly
- `count_jsonl` matches `len(load_jsonl(path))`

---

- [ ] **Unit 2: Update nugget extraction writers**

**Goal:** Switch `src/nuggets/extract.py` and `src/nuggets/unified.py` to write `.jsonl` output.

**Requirements:** R1, R5

**Dependencies:** Unit 1

**Files:**
- Modify: `src/nuggets/extract.py`
- Modify: `src/nuggets/unified.py`

**Approach:**
- **extract.py**: Replace the envelope-building block (~line 450-460) with `save_jsonl(nuggets, out_path, removed=removed)`. Change output filename from `.json` to `.jsonl`. For empty results (0 chunks), write an empty file as a "done" marker.
- **unified.py — write path**: Replace `save_json(result, path)` in `_on_result` (~line 623) with `save_jsonl(result["nuggets"], path, removed=result.get("removed"))`. Update the empty-result write (~line 596-601) to write an empty `.jsonl` file.
- **unified.py — `_select_oldest`** (~line 687-712): Change `.json` extension check to `.jsonl` in the directory scan.
- **unified.py — reprocess/review input reading** (~lines 604-606): Change `load_json(os.path.join(src_dir, f"{paper_id}.json"))` and `.get("nuggets", [])` to `load_jsonl(os.path.join(src_dir, f"{paper_id}.jsonl"))`. Filter `_removed` if present: `[n for n in load_jsonl(path) if not n.get("_removed")]`.
- Progress counters (`counters["nuggets"]`, `counters["removed"]`) are computed from the returned result dict, not from the file — no change needed for logging.

**Patterns to follow:**
- Current envelope construction in extract.py and unified.py — replace `save_json(result, path)` with `save_jsonl(result["nuggets"], path, removed=result.get("removed"))`

**Test scenarios:**
- Happy path: extraction produces `.jsonl` file with one nugget per line
- Happy path: unified produces `.jsonl` with quality-scored nuggets + removed nuggets flagged
- Happy path: `_select_oldest` finds `.jsonl` files in unified_dir
- Happy path: reprocess mode reads `.jsonl` input, filters `_removed` entries
- Edge case: paper with 0 valid chunks produces empty `.jsonl` file

**Verification:**
- Running extraction on a test paper produces `corpus/nuggets/{paper_id}.jsonl`
- Each line is valid JSON; no envelope wrapper
- `--oldest` flag works with `.jsonl` files

---

- [ ] **Unit 3: Update skip-check functions**

**Goal:** Make skip-check functions look for `.jsonl` files instead of `.json`.

**Requirements:** R4

**Dependencies:** Unit 2

**Files:**
- Modify: `src/nuggets/extract.py` (`_nugget_done`)
- Modify: `src/nuggets/unified.py` (`_done`)

**Approach:**
- Replace JSON-parsing skip checks with: check if `.jsonl` file exists via `os.path.exists(path)`.
- For extract.py's `_nugget_done`: check both `nugget_dir` and `unified_dir` for `.jsonl` files.
- For unified.py's `_done`: check `unified_dir` for `.jsonl`.
- **Semantic change**: Current behavior reprocesses papers with `num_nuggets == 0`. New behavior treats any existing file (even empty) as "done." This is intentional — papers with no extractable content won't produce nuggets on re-extraction either. To force reprocessing, delete the output file (consistent with the documented reprocessing pattern in CLAUDE.md).

**Patterns to follow:**
- `src/utils.py:already_processed()` — checks file existence + size

**Test scenarios:**
- Happy path: existing `.jsonl` file (non-empty) → skip
- Happy path: no file → process
- Edge case: empty `.jsonl` file (0 nuggets from 0-chunk paper) → skip (deliberate change from current reprocess-on-zero)

**Verification:**
- Pipeline correctly skips already-processed papers
- Pipeline processes new papers
- Papers with empty output files are not reprocessed every run

---

- [ ] **Unit 4: Update embedder (load_all_nuggets + output)**

**Goal:** Switch the main nugget aggregator to read `.jsonl` files and write JSONL output.

**Requirements:** R2, R3

**Dependencies:** Unit 2

**Files:**
- Modify: `src/embed/embedder.py`

**Approach:**
- In `load_all_nuggets()`: iterate `.jsonl` files instead of `.json`. Use `load_jsonl()` to get nugget list directly (no `.get("nuggets", [])` unwrapping needed). Filter `_removed`: `[n for n in load_jsonl(path) if not n.get("_removed")]`.
- Legacy augmented merge logic: the embedder currently merges from `augmented_dir` using `.get("improved", [])` and `.get("gap_filled", [])`. Since augment.py is legacy/out-of-scope, keep this reader path for `.json` files in `augmented_dir` unchanged. If `augmented_dir` doesn't exist or is empty, the code already handles that.
- Output: write `nuggets_with_embeddings.jsonl` using `save_jsonl()`. After embedding, each nugget gets `embedding_idx` added (integer, 0-indexed, matches row in `embeddings.npy`). The aggregated file contains only non-removed nuggets — `_removed` entries are filtered during `load_all_nuggets`, so they never reach the aggregated output.
- Update `_load_existing_kb()` to read `.jsonl` for incremental mode. On first post-migration run, the `.jsonl` file won't exist; `_load_existing_kb` returns `(None, None)`, `run_embedding` sets `incremental = False`, and a full re-embed runs.

**Aggregated file format contract (Unit 4 → Unit 5):**

`nuggets_with_embeddings.jsonl` — one nugget per line. Each line is a JSON object with at minimum these fields (consumed by `store/kb.py:build_sqlite` and `build_chromadb`):

| Field | Type | Used by |
|-------|------|---------|
| `nugget_id` | str | SQLite PK, ChromaDB ID |
| `paper_id` | str | SQLite FK, ChromaDB metadata |
| `question` | str | SQLite, ChromaDB document, FTS5 |
| `answer` | str | SQLite, ChromaDB document, FTS5 |
| `type` | str | SQLite, ChromaDB metadata |
| `section` | str | SQLite, ChromaDB metadata |
| `source_chunk` | int | SQLite |
| `embedding_idx` | int | Row index into `embeddings.npy` |
| `quality` | dict (optional) | SQLite (`overall` score, `flagged` derivation) |
| `confidence` | str (optional) | SQLite |
| `paper_title` | str (optional) | ChromaDB metadata |
| `paper_authors` | list (optional) | ChromaDB metadata |
| `paper_year` | int (optional) | ChromaDB metadata |

No `_removed` entries appear in this file. The companion `embeddings.npy` has shape `(N, D)` where `N == line count` and row `i` corresponds to the nugget on line `i`.

**Patterns to follow:**
- Current `load_all_nuggets` iteration pattern — same directory scanning, just different file extension and loader

**Test scenarios:**
- Happy path: loads nuggets from unified `.jsonl` files, skips `_removed` entries
- Happy path: produces `nuggets_with_embeddings.jsonl` with correct `embedding_idx`
- Edge case: mix of papers — unified `.jsonl` + legacy augmented `.json` both loaded
- Edge case: incremental mode, first run post-migration → full rebuild (no `.jsonl` KB yet)
- Edge case: incremental mode, second run post-migration → reads `.jsonl` KB, appends only new

**Verification:**
- `nuggets_with_embeddings.jsonl` has one nugget per line with `embedding_idx`
- Nugget count matches sum of all per-paper files (excluding removed)

---

- [ ] **Unit 5: Update store stage**

**Goal:** Switch `src/store/kb.py` to read the JSONL aggregated file.

**Requirements:** R3

**Dependencies:** Unit 4

**Files:**
- Modify: `src/store/kb.py`

**Approach:**
- Change `load_json(nug_path)` to `load_jsonl(nug_path)` for the aggregated file.
- Update path from `nuggets_with_embeddings.json` to `nuggets_with_embeddings.jsonl`.
- Quality score reading from `corpus/nuggets_quality/` stays as-is (JSON format, legacy stage out of scope). The unified pipeline inlines quality scores per-nugget, which store.py already reads from the `quality` key on each nugget.

**Patterns to follow:**
- Current `run_build` loading pattern — same field access on individual nuggets

**Test scenarios:**
- Happy path: store builds ChromaDB + SQLite from `.jsonl` aggregated file
- Happy path: inlined quality scores (from unified pipeline) loaded correctly from per-nugget `quality` key

**Verification:**
- `python -m src.store` completes without error
- SQLite `nuggets` table has expected row count

## System-Wide Impact

- **Interaction graph:** Writers (extract, unified) → readers (embed, store). Units 2-5 form the critical path.
- **Error propagation:** A half-migrated pipeline (some stages writing `.json`, others expecting `.jsonl`) will silently produce empty results. Units 2 and 3 must be deployed together per module to avoid desynchronization.
- **State lifecycle risks:** Existing `corpus/nuggets/*.json` and `corpus/nuggets_unified/*.json` files will be ignored after migration. Users must delete old files and rerun extraction, or run a full pipeline rebuild. The pipeline will not corrupt old files.
- **API surface parity:** MCP server reads from SQLite/ChromaDB only — unaffected.
- **Unchanged invariants:** The nugget object schema (question, answer, type, paper_id, section, etc.) does not change. Only the container format changes. Legacy stages (quality.py, augment.py, enrich.py) continue to read/write JSON — they are not broken, just not migrated.

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| Half-migrated state (old `.json` + new `.jsonl` coexist) | Deploy Units 2+3 together per module. Document that old files must be deleted before rerunning. |
| `_removed` flag in per-paper files | Only embedder filters these. Unified reprocess mode also filters when reading input. Store sees already-filtered aggregated file. |
| Large reprocessing cost after migration | Embed + store are full rebuilds anyway. Only nugget extraction needs reprocessing if old files are deleted. With 1500+ papers on HPC, this is a ~2-4 hour GPU job. |
| Incremental embed finds no `.jsonl` KB on first post-migration run | `_load_existing_kb` returns `(None, None)`, `run_embedding` sets `incremental = False`, full re-embed runs. One-time cost. |
| Legacy stages break if someone runs them | Out of scope — they are documented as unused. Would need separate migration if revived. |

## Sources & References

- Related code: `src/utils.py`, `src/nuggets/extract.py`, `src/nuggets/unified.py`, `src/embed/embedder.py`, `src/store/kb.py`
- Out of scope: `src/code_extract/nuggets.py`, `src/nuggets/quality.py`, `src/nuggets/augment.py`, `src/nuggets/enrich.py`
