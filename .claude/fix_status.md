# P0/P1 Fix Status

## Completed

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

## All P0/P1 fixes complete.

## Fixes applied

| # | Priority | Issue | File(s) |
|---|----------|-------|---------|
| ~~1~~ | ~~P0~~ | ~~Augmented nuggets never embedded~~ | ~~embedder.py~~ |
| ~~2~~ | ~~P0~~ | ~~O(n^2) dedup with no bound~~ | ~~extract.py~~ |
| ~~3~~ | ~~P0~~ | ~~download_pdf timeout ignored~~ | ~~fetch.py~~ |
| ~~4~~ | ~~P1~~ | ~~No README.md~~ | ~~README.md~~ |
| ~~5~~ | ~~P1~~ | ~~S2 enrichment accepts wrong papers~~ | ~~enrich.py~~ |
| ~~6~~ | ~~P1~~ | ~~SQLite connection leak on init error~~ | ~~query.py~~ |
| ~~7~~ | ~~P1~~ | ~~OCR fallback doesn't do OCR~~ | ~~pdf_to_text.py~~ |
| ~~8~~ | ~~P1~~ | ~~Embedding progress log broken~~ | ~~embedder.py~~ |
| ~~9~~ | ~~P1~~ | ~~ChromaDB silently swallows delete errors~~ | ~~kb.py~~ |
| ~~10~~ | ~~P1~~ | ~~Embedding text may exceed max_model_len~~ | ~~config.yaml, embedder.py~~ |

## Current KB state

The current `kb/` contains **144,764 nuggets** indexed from **base nuggets only** (`corpus/nuggets/`). The augmented nuggets in `corpus/nuggets_augmented/` (improved + gap-filled) have never been embedded or indexed. A re-run is required to bring the KB up to date with fix #1.

## Required: Re-embed and rebuild

Run the embed + store pipeline to index augmented nuggets (fixes #1, #8, #9, #10 take effect):

```bash
sbatch slurm/embed_nuggets.slurm
```

This runs both `python -m src.embed` and `python -m src.store` in sequence.

Alternatively, to also re-run quality + augmentation (if augmented data needs refreshing):

```bash
sbatch slurm/reprocess_pipeline.slurm
```

## Verification after re-run

Check that augmented nuggets are now included:

```bash
# Total nugget count should be > 144,764
python -c "from src.utils import load_json; n = load_json('kb/nuggets_with_embeddings.json'); print(f'Total nuggets: {len(n)}')"

# Count gap-filled nuggets (should be > 0)
python -c "from src.utils import load_json; n = load_json('kb/nuggets_with_embeddings.json'); gap = [x for x in n if '_gap_' in x.get('nugget_id', '')]; print(f'Gap-filled nuggets: {len(gap)}')"

# Count improved nuggets (should be > 0)
python -c "from src.utils import load_json; n = load_json('kb/nuggets_with_embeddings.json'); imp = [x for x in n if x.get('improved')]; print(f'Improved nuggets: {len(imp)}')"

# Verify embedding matrix matches nugget count
python -c "import numpy as np; e = np.load('kb/embeddings.npy'); print(f'Embedding shape: {e.shape}')"

# Test a query
python -m src.query "event camera object detection" -n 5
```

## Not addressed: P2/P3 items

The following lower-priority items from the audit were **not fixed** in this pass. See the full plan at `~/.claude/plans/reactive-forging-dahl.md` for details.

| # | Priority | Issue |
|---|----------|-------|
| 11 | P2 | Duplicate LLM client init across 3 files |
| 12 | P2 | Unused config key `max_chunk_chars` |
| 13 | P2 | Inconsistent skip-if-done checks across stages |
| 14 | P2 | Augmented nuggets not enriched with paper metadata |
| 15 | P2 | `multi_query` skewed per-query distribution |
| 16 | P2 | Python 3.12+ f-string syntax limits portability |
| 17 | P2 | `pdf_dir: ~` not expanded by `os.path.join` |
| 18 | P3 | No test suite |
| 19 | P3 | `save_json` atomic write races on NFS |
| 20 | P3 | `repair_json` greedy regex edge case |
| 21 | P3 | No graceful shutdown on SIGTERM |
| 22 | P3 | Python 3.12+ f-string in kb.py |
| 23 | P3 | Nugget/embedding count not validated in store |
| 24 | P3 | vLLM `--convert embed` flag may be version-dependent |
| 25 | P3 | Standalone utility scripts undocumented |
