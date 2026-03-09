# Codebase Audit Fix Status

## All P0/P1 fixes complete.

- **P0 #1: Augmented nuggets never embedded** — `src/embed/embedder.py` modified. `load_all_nuggets` now merges improved/gap-filled nuggets from `augmented_dir` before embedding. Backward compatible.
- **P0 #2: O(n^2) dedup with no bound** — `src/nuggets/extract.py:119-135` modified. Word sets now cached in parallel list instead of recomputed per comparison. `augment.py` already used the efficient pattern.
- **P0 #3: download_pdf timeout ignored** — `src/acquire/fetch.py:10-23` modified. Replaced `urlretrieve` with `urlopen(url, timeout=timeout)` so the timeout parameter is actually honored.
- **P1 #4: No README.md** — Created `README.md` with pipeline overview, setup, usage (module + SLURM), configuration, and directory structure.
- **P1 #5: S2 enrichment accepts wrong papers** — `src/acquire/enrich.py:22` modified. Now checks title similarity (>=0.75 via `SequenceMatcher`) across top 3 results before accepting; rejects all if none match.
- **P1 #6: SQLite connection leak on init error** — `src/query.py` modified. Added `__enter__`/`__exit__` for context manager support, init `self.db = None` early so `close()` is safe if ChromaDB fails, and `close()` guards against None.
- **P1 #7: OCR fallback doesn't do OCR** — `src/extract/pdf_to_text.py:33-45` modified. Removed misleading "OCR fallback" that just re-extracted text with a different flag. Now logs a warning for scanned/image-only pages. `ocr_fallback` config key was never read by code.
- **P1 #8: Embedding progress log broken** — `src/embed/embedder.py:137` modified. Changed modulo condition from `(start + batch_size) % (batch_size * 10)` to `batch_num % 10 == 0`, so progress logs every 10 batches as intended.
- **P1 #9: ChromaDB silently swallows delete errors** — `src/store/kb.py:15-18` modified. Narrowed bare `except Exception` to `except ValueError` so only "collection not found" is caught; real errors (connection, permissions) now propagate.
- **P1 #10: Embedding text may exceed max_model_len** — `src/embed/embedder.py` + `config.yaml` modified. `format_nugget_text` now accepts `max_tokens` and truncates via tiktoken. `run_embedding` reads `max_model_len` from config and passes it through. Bumped config from 512 to 1024 to match SLURM scripts.

## P2/P3 fixes complete.

- **P2 #11: Duplicate LLM client init** — Extracted `make_llm_client(cfg)` to `src/utils.py`. Replaced duplicated init blocks in `extract.py`, `quality.py`, `augment.py`.
- **P2 #12: Unused config key `max_chunk_chars`** — Removed from `config.yaml`.
- **P2 #14: Augmented nuggets not enriched with paper metadata** — `src/nuggets/enrich.py` modified. Extracted `_enrich_dir()` helper; now processes both `nugget_dir` and `augmented_dir`, iterating all nugget list keys (`nuggets`, `improved`, `gap_filled`).
- **P2 #15: `multi_query` skewed per-query distribution** — `src/query.py:130` modified. Changed from `max(n_results, n_results // len(queries) + 5)` to `max(5, ceil(n_results / len(queries)) + 3)` for balanced distribution.
- **P2 #16 + P3 #22: Python 3.12+ f-string nesting** — Fixed in `src/store/kb.py:33` and `src/acquire/fetch.py:160,187`. Replaced nested `"` inside f-strings with `'` or extracted to variables.
- **P2 #17: `pdf_dir: ~` not expanded** — `src/utils.py:load_config` now calls `os.path.expanduser()` on all path values containing `~`.
- **P3 #23: Nugget/embedding count not validated in store** — `src/store/kb.py:run_build` now checks `len(nuggets) == embeddings.shape[0]` and aborts with error message on mismatch.

## Not fixed (accepted / deferred)

| # | Priority | Issue | Reason |
|---|----------|-------|--------|
| 13 | P2 | Inconsistent skip-if-done checks | By design — each stage has valid different semantics |
| 18 | P3 | No test suite | Scope — would need dedicated effort |
| 19 | P3 | `save_json` atomic write races on NFS | Low risk — `os.replace` in same dir is safe enough |
| 20 | P3 | `repair_json` greedy regex | Not a real bug — greedy `[.*]` correctly finds outer array, `json.loads` validates |
| 21 | P3 | No graceful shutdown on SIGTERM | Low impact — partial files are skipped on re-run |
| 24 | P3 | vLLM `--convert embed` flag version-dependent | Documentation only — pin vLLM version in requirements |
| 25 | P3 | Standalone utility scripts undocumented | Low priority |

## Fixes applied (full table)

| # | Priority | Issue | File(s) | Status |
|---|----------|-------|---------|--------|
| ~~1~~ | ~~P0~~ | ~~Augmented nuggets never embedded~~ | ~~embedder.py~~ | Done |
| ~~2~~ | ~~P0~~ | ~~O(n^2) dedup with no bound~~ | ~~extract.py~~ | Done |
| ~~3~~ | ~~P0~~ | ~~download_pdf timeout ignored~~ | ~~fetch.py~~ | Done |
| ~~4~~ | ~~P1~~ | ~~No README.md~~ | ~~README.md~~ | Done |
| ~~5~~ | ~~P1~~ | ~~S2 enrichment accepts wrong papers~~ | ~~enrich.py~~ | Done |
| ~~6~~ | ~~P1~~ | ~~SQLite connection leak on init error~~ | ~~query.py~~ | Done |
| ~~7~~ | ~~P1~~ | ~~OCR fallback doesn't do OCR~~ | ~~pdf_to_text.py~~ | Done |
| ~~8~~ | ~~P1~~ | ~~Embedding progress log broken~~ | ~~embedder.py~~ | Done |
| ~~9~~ | ~~P1~~ | ~~ChromaDB silently swallows delete errors~~ | ~~kb.py~~ | Done |
| ~~10~~ | ~~P1~~ | ~~Embedding text may exceed max_model_len~~ | ~~config.yaml, embedder.py~~ | Done |
| ~~11~~ | ~~P2~~ | ~~Duplicate LLM client init~~ | ~~utils.py, extract.py, quality.py, augment.py~~ | Done |
| ~~12~~ | ~~P2~~ | ~~Unused config key max_chunk_chars~~ | ~~config.yaml~~ | Done |
| 13 | P2 | Inconsistent skip-if-done checks | — | Accepted |
| ~~14~~ | ~~P2~~ | ~~Augmented nuggets not enriched~~ | ~~enrich.py~~ | Done |
| ~~15~~ | ~~P2~~ | ~~multi_query skewed distribution~~ | ~~query.py~~ | Done |
| ~~16~~ | ~~P2~~ | ~~Python 3.12+ f-string in fetch.py~~ | ~~fetch.py~~ | Done |
| ~~17~~ | ~~P2~~ | ~~pdf_dir ~ not expanded~~ | ~~utils.py~~ | Done |
| 18 | P3 | No test suite | — | Deferred |
| 19 | P3 | save_json NFS races | — | Accepted |
| 20 | P3 | repair_json greedy regex | — | Not a bug |
| 21 | P3 | No graceful shutdown | — | Accepted |
| ~~22~~ | ~~P3~~ | ~~Python 3.12+ f-string in kb.py~~ | ~~kb.py~~ | Done |
| ~~23~~ | ~~P3~~ | ~~Nugget/embedding count not validated~~ | ~~kb.py~~ | Done |
| 24 | P3 | vLLM flag version-dependent | — | Deferred |
| 25 | P3 | Utility scripts undocumented | — | Deferred |

## KB state (post re-embed)

The KB contains **150,728 nuggets** (28,829 improved + 5,964 gap-filled merged from augmented pipeline). Re-embed completed successfully via `sbatch slurm/embed_nuggets.slurm`.
