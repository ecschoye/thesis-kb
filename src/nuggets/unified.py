"""Unified nugget pipeline: extract → quality → augment in one per-paper pass.

Keeps chunks in memory, runs all three stages sequentially per paper,
saves one output JSON per paper with inlined quality scores and audit trail.

Usage:
    python -m src.nuggets.unified -c config.yaml                # full pipeline (new papers)
    python -m src.nuggets.unified -c config.yaml --reprocess    # quality+augment on raw nuggets
    python -m src.nuggets.unified -c config.yaml --regenerate   # force re-extract from chunks
    python -m src.nuggets.unified -c config.yaml --review       # review existing unified nuggets
    python -m src.nuggets.unified -c config.yaml --oldest 500   # review 500 oldest (implies --review)
"""
import os
import json
import sys
import time
import argparse
import threading
import itertools
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils import load_config, load_json, save_json, save_jsonl, load_jsonl, make_llm_clients

# Reuse existing stage implementations
from src.nuggets.extract import _process_chunk, _deduplicate_nuggets
from src.nuggets.quality import rate_nugget_batch
from src.nuggets.augment import (
    improve_nugget,
    gapfill_chunk,
    _dedup_against_existing,
    _is_reference_chunk,
)


def _get_paper_title(nuggets, paper_id):
    """Best-effort paper title from nuggets (background type mentioning title)."""
    for n in nuggets:
        if n.get("type") == "background":
            q = n.get("question", "").lower()
            if "title" in q or "author" in q:
                # Title is usually in the answer
                return n.get("answer", paper_id)[:200]
    return paper_id


def _process_paper_unified(
    client, paper_id, chunks, model, ext_cfg, qcfg, acfg, ucfg,
    extra_body=None, worker_id=0, print_lock=None, counters=None,
    max_model_len=8192, paper_meta=None, self_score=False,
):
    """Full extract→quality→augment pipeline for one paper, all in memory.

    When self_score=True, extraction includes self-assessed quality scores
    and the separate quality rating stage (Step 3) is skipped.

    Returns output dict ready for save_json.
    """
    flag_threshold = ucfg.get("flag_threshold", 2)
    improve_threshold = ucfg.get("improve_threshold", 2)
    gap_max_nuggets = acfg.get("gap_max_nuggets", 2)
    gap_min_tokens = acfg.get("gap_min_tokens", 100)
    gap_skip_sections = set(acfg.get("gap_skip_sections", [
        "references", "acknowledgments", "bibliography",
    ]))

    short_id = paper_id[:30]

    def _log(msg):
        if print_lock and counters:
            with print_lock:
                sys.stderr.write(f"\r\033[K  [{counters['done']}/{counters['total']}] {msg}\n")
                sys.stderr.flush()

    # ── Step 1: Extract nuggets from chunks (parallel) ────────────────────
    temp = ext_cfg.get("temperature", 0.1)
    max_tok = ext_cfg.get("max_tokens", 3000)
    max_retries = ext_cfg.get("max_retries", 3)
    retry_delay = ext_cfg.get("retry_base_delay", 2.0)

    chunk_by_id = {}
    for c in chunks:
        chunk_by_id[c["chunk_id"]] = c

    # Fire all chunk extractions concurrently — no sequential prior_questions
    # dependency. Post-hoc dedup handles duplicates instead.
    def _extract_one(ci_c):
        ci, c = ci_c
        try:
            return _process_chunk(
                client, c, model, temp, max_tok,
                max_retries, retry_delay, paper_id, extra_body,
                prior_questions=None,
                max_model_len=max_model_len,
                paper_meta=paper_meta,
                self_score=self_score,
            )
        except Exception as e:
            _log(f"  WARN {short_id} chunk {ci}: {e}")
            return []

    all_raw_nuggets = []
    with ThreadPoolExecutor(max_workers=min(len(chunks), 8)) as chunk_executor:
        futures = chunk_executor.map(_extract_one, enumerate(chunks))
        for chunk_nuggets in futures:
            all_raw_nuggets.extend(chunk_nuggets)

    # ── Step 2: Deduplicate (post-hoc, replaces sequential prior_questions) ──
    deduped = _deduplicate_nuggets(all_raw_nuggets, paper_id)
    for n in deduped:
        n["origin"] = "extracted"

    _log(f"{short_id}: {len(deduped)} nuggets extracted (deduped from {len(all_raw_nuggets)})")

    if not deduped:
        return {
            "paper_id": paper_id,
            "num_nuggets": 0, "num_removed": 0,
            "num_improved": 0, "num_gap_filled": 0,
            "quality_summary": {},
            "nuggets": [], "removed": [],
        }

    # ── Step 3: Quality rating ───────────────────────────────────────────
    if self_score:
        # Self-scored nuggets already have inlined quality — skip LLM rating
        _log(f"{short_id}: using self-assessed quality scores (skipping separate rating)")
        # Ensure all nuggets have quality dict (fallback for any that didn't parse scores)
        for n in deduped:
            if "quality" not in n:
                n["quality"] = {
                    "relevance": 3, "specificity": 3, "self_contained": 3,
                    "type_accuracy": 3, "coherence": 3, "thesis_relevance": 3,
                    "overall": 3, "flags": ["self_score_missing"],
                }
    else:
        paper_title = _get_paper_title(deduped, paper_id)
        batch_size = qcfg.get("batch_size", 5)
        quality_by_id = {}

        for i in range(0, len(deduped), batch_size):
            batch = deduped[i:i + batch_size]
            results, err = rate_nugget_batch(
                client, batch, model, paper_title, paper_id, qcfg,
                max_model_len=max_model_len, extra_body=extra_body,
            )
            if results:
                for r in results:
                    quality_by_id[r["nugget_id"]] = r
            else:
                _log(f"  WARN {short_id} quality batch {i // batch_size}: {err}")
                for n in batch:
                    quality_by_id[n["nugget_id"]] = {
                        "nugget_id": n["nugget_id"],
                        "relevance": 0, "specificity": 0, "self_contained": 0,
                        "type_accuracy": 0, "coherence": 0, "thesis_relevance": 0,
                        "overall": 0, "flags": ["batch_failed"],
                    }

        # Inline quality scores onto nuggets
        for n in deduped:
            nid = n["nugget_id"]
            q = quality_by_id.get(nid, {})
            n["quality"] = {
                "relevance": q.get("relevance", 0),
                "specificity": q.get("specificity", 0),
                "self_contained": q.get("self_contained", 0),
                "type_accuracy": q.get("type_accuracy", 0),
                "coherence": q.get("coherence", 0),
                "thesis_relevance": q.get("thesis_relevance", 0),
                "overall": q.get("overall", 0),
                "flags": q.get("flags", []),
            }

    # ── Step 3b: Bail out if scoring failed entirely ──────────────────────
    all_scores = [n["quality"]["overall"] for n in deduped]
    if all_scores and all(s == 0 for s in all_scores):
        _log(f"{short_id}: scoring failed for all nuggets — keeping extracted nuggets as-is")
        for i, n in enumerate(deduped):
            n["nugget_id"] = f"{paper_id}_{i}"
        return {
            "paper_id": paper_id,
            "num_nuggets": len(deduped),
            "num_removed": 0,
            "num_improved": 0,
            "num_gap_filled": 0,
            "quality_summary": {"scoring_failed": True},
            "nuggets": deduped,
            "removed": [],
        }

    # ── Step 4: Improve weak nuggets ─────────────────────────────────────
    improved_count = 0
    removed = []
    keep = []

    for n in deduped:
        overall = n["quality"]["overall"]
        if overall > flag_threshold:
            keep.append(n)
            continue

        # Attempt improvement if score > 0 (0 = rating failed)
        if overall > 0:
            chunk = chunk_by_id.get(n.get("source_chunk"))
            if chunk:
                result = improve_nugget(
                    client, n, n["quality"], chunk["text"], model, acfg,
                    extra_body=extra_body,
                )
                if result.get("improved"):
                    n["question"] = result["question"]
                    n["answer"] = result["answer"]
                    n["type"] = result.get("type", n["type"])
                    n["origin"] = "improved"
                    n["quality"]["overall"] = improve_threshold + 1  # synthetic pass
                    n["quality"]["flags"] = []
                    improved_count += 1
                    keep.append(n)
                    continue

        # Still weak after improvement attempt (or no chunk to improve with)
        removed.append({
            "nugget_id": n["nugget_id"],
            "question": n["question"],
            "answer": n["answer"],
            "type": n["type"],
            "quality": n["quality"],
            "removal_reason": "overall_score_below_threshold",
        })

    _log(f"{short_id}: {improved_count} improved, {len(removed)} removed, {len(keep)} kept")

    # ── Step 5: Gap-fill sparse chunks ───────────────────────────────────
    # Build surviving nuggets-per-chunk map
    nuggets_by_chunk = defaultdict(list)
    for n in keep:
        nuggets_by_chunk[n.get("source_chunk")].append(n)

    all_gap_filled = []
    for chunk_id, chunk in chunk_by_id.items():
        section = chunk.get("section", "")
        if section in gap_skip_sections:
            continue
        if chunk.get("token_count", 0) < gap_min_tokens:
            continue
        if _is_reference_chunk(chunk["text"]):
            continue
        existing = nuggets_by_chunk.get(chunk_id, [])
        if len(existing) > gap_max_nuggets:
            continue

        new_nuggets = gapfill_chunk(client, chunk["text"], existing, model, acfg,
                                    extra_body=extra_body)
        for gn in new_nuggets:
            gn["source_chunk"] = chunk_id
            gn["section"] = section
            gn["pages"] = chunk.get("pages", [])
            gn["paper_id"] = paper_id
            gn["origin"] = "gap_filled"
            gn["quality"] = {
                "relevance": 0, "specificity": 0, "self_contained": 0,
                "type_accuracy": 0, "coherence": 0, "thesis_relevance": 0,
                "overall": improve_threshold + 1,  # trusted — passes filter
                "flags": ["unrated_gap_fill"],
            }
        all_gap_filled.extend(new_nuggets)

    # Dedup gap-filled against existing kept nuggets
    all_gap_filled = _dedup_against_existing(all_gap_filled, keep)

    # Assign IDs to gap-filled nuggets
    for i, gn in enumerate(all_gap_filled):
        gn["nugget_id"] = f"{paper_id}_gap_{i}"

    _log(f"{short_id}: {len(all_gap_filled)} gap-filled nuggets")

    # ── Step 6: Final merge + renumber ───────────────────────────────────
    final = keep + all_gap_filled

    # Re-number nugget_ids sequentially
    for i, n in enumerate(final):
        n["nugget_id"] = f"{paper_id}_{i}"

    # ── Step 7: Build output ─────────────────────────────────────────────
    valid_scores = [n["quality"]["overall"] for n in final if n["quality"]["overall"] > 0]
    score_dist = defaultdict(int)
    for s in valid_scores:
        score_dist[str(s)] += 1

    return {
        "paper_id": paper_id,
        "num_nuggets": len(final),
        "num_removed": len(removed),
        "num_improved": improved_count,
        "num_gap_filled": len(all_gap_filled),
        "quality_summary": {
            "mean_overall": round(sum(valid_scores) / len(valid_scores), 2) if valid_scores else 0,
            "num_flagged": len(removed),
            "score_distribution": dict(score_dist),
        },
        "nuggets": final,
        "removed": removed,
    }


def _process_paper_reprocess(
    client, paper_id, nuggets, chunks, model, qcfg, acfg, ucfg,
    extra_body=None, print_lock=None, counters=None,
    max_model_len=8192,
):
    """Quality + augment on existing nuggets (skip extraction)."""
    flag_threshold = ucfg.get("flag_threshold", 2)
    improve_threshold = ucfg.get("improve_threshold", 2)
    gap_max_nuggets = acfg.get("gap_max_nuggets", 2)
    gap_min_tokens = acfg.get("gap_min_tokens", 100)
    gap_skip_sections = set(acfg.get("gap_skip_sections", [
        "references", "acknowledgments", "bibliography",
    ]))

    short_id = paper_id[:30]

    def _log(msg):
        if print_lock and counters:
            with print_lock:
                sys.stderr.write(f"\r\033[K  [{counters['done']}/{counters['total']}] {msg}\n")
                sys.stderr.flush()

    chunk_by_id = {c["chunk_id"]: c for c in chunks}

    # Set origin for existing nuggets
    for n in nuggets:
        n["origin"] = "extracted"

    if not nuggets:
        return {
            "paper_id": paper_id,
            "num_nuggets": 0, "num_removed": 0,
            "num_improved": 0, "num_gap_filled": 0,
            "quality_summary": {},
            "nuggets": [], "removed": [],
        }

    # Quality rating
    paper_title = _get_paper_title(nuggets, paper_id)
    batch_size = qcfg.get("batch_size", 5)
    quality_by_id = {}

    for i in range(0, len(nuggets), batch_size):
        batch = nuggets[i:i + batch_size]
        results, err = rate_nugget_batch(
            client, batch, model, paper_title, paper_id, qcfg,
            max_model_len=max_model_len, extra_body=extra_body,
        )
        if results:
            for r in results:
                quality_by_id[r["nugget_id"]] = r
        else:
            _log(f"  WARN {short_id} quality batch {i // batch_size}: {err}")
            for n in batch:
                quality_by_id[n["nugget_id"]] = {
                    "nugget_id": n["nugget_id"],
                    "relevance": 0, "specificity": 0, "self_contained": 0,
                    "type_accuracy": 0, "coherence": 0, "thesis_relevance": 0,
                    "overall": 0, "flags": ["batch_failed"],
                }

    for n in nuggets:
        q = quality_by_id.get(n["nugget_id"], {})
        n["quality"] = {
            "relevance": q.get("relevance", 0),
            "specificity": q.get("specificity", 0),
            "self_contained": q.get("self_contained", 0),
            "type_accuracy": q.get("type_accuracy", 0),
            "coherence": q.get("coherence", 0),
            "thesis_relevance": q.get("thesis_relevance", 0),
            "overall": q.get("overall", 0),
            "flags": q.get("flags", []),
        }

    # If ALL nuggets have overall=0, scoring failed entirely — keep originals unchanged
    all_scores = [n["quality"]["overall"] for n in nuggets]
    if all_scores and all(s == 0 for s in all_scores):
        _log(f"{short_id}: scoring failed for all nuggets — skipping review, keeping originals")
        return {
            "paper_id": paper_id,
            "num_nuggets": len(nuggets),
            "num_removed": 0,
            "num_improved": 0,
            "num_gap_filled": 0,
            "quality_summary": {"scoring_failed": True},
            "nuggets": nuggets,
            "removed": [],
        }

    # Improve + filter
    improved_count = 0
    removed = []
    keep = []

    for n in nuggets:
        overall = n["quality"]["overall"]
        if overall > flag_threshold:
            keep.append(n)
            continue
        if overall > 0:
            chunk = chunk_by_id.get(n.get("source_chunk"))
            if chunk:
                result = improve_nugget(
                    client, n, n["quality"], chunk["text"], model, acfg,
                    extra_body=extra_body,
                )
                if result.get("improved"):
                    n["question"] = result["question"]
                    n["answer"] = result["answer"]
                    n["type"] = result.get("type", n["type"])
                    n["origin"] = "improved"
                    n["quality"]["overall"] = improve_threshold + 1
                    n["quality"]["flags"] = []
                    improved_count += 1
                    keep.append(n)
                    continue
        removed.append({
            "nugget_id": n["nugget_id"],
            "question": n["question"],
            "answer": n["answer"],
            "type": n["type"],
            "quality": n["quality"],
            "removal_reason": "overall_score_below_threshold",
        })

    # Gap-fill
    nuggets_by_chunk = defaultdict(list)
    for n in keep:
        nuggets_by_chunk[n.get("source_chunk")].append(n)

    all_gap_filled = []
    for chunk_id, chunk in chunk_by_id.items():
        section = chunk.get("section", "")
        if section in gap_skip_sections:
            continue
        if chunk.get("token_count", 0) < gap_min_tokens:
            continue
        if _is_reference_chunk(chunk["text"]):
            continue
        existing = nuggets_by_chunk.get(chunk_id, [])
        if len(existing) > gap_max_nuggets:
            continue
        new_nuggets = gapfill_chunk(client, chunk["text"], existing, model, acfg,
                                    extra_body=extra_body)
        for gn in new_nuggets:
            gn["source_chunk"] = chunk_id
            gn["section"] = section
            gn["pages"] = chunk.get("pages", [])
            gn["paper_id"] = paper_id
            gn["origin"] = "gap_filled"
            gn["quality"] = {
                "relevance": 0, "specificity": 0, "self_contained": 0,
                "type_accuracy": 0, "coherence": 0, "thesis_relevance": 0,
                "overall": improve_threshold + 1,
                "flags": ["unrated_gap_fill"],
            }
        all_gap_filled.extend(new_nuggets)

    all_gap_filled = _dedup_against_existing(all_gap_filled, keep)
    for i, gn in enumerate(all_gap_filled):
        gn["nugget_id"] = f"{paper_id}_gap_{i}"

    final = keep + all_gap_filled
    for i, n in enumerate(final):
        n["nugget_id"] = f"{paper_id}_{i}"

    valid_scores = [n["quality"]["overall"] for n in final if n["quality"]["overall"] > 0]
    score_dist = defaultdict(int)
    for s in valid_scores:
        score_dist[str(s)] += 1

    _log(f"{short_id}: {len(keep)} kept, {improved_count} improved, {len(removed)} removed, {len(all_gap_filled)} gap-filled")

    return {
        "paper_id": paper_id,
        "num_nuggets": len(final),
        "num_removed": len(removed),
        "num_improved": improved_count,
        "num_gap_filled": len(all_gap_filled),
        "quality_summary": {
            "mean_overall": round(sum(valid_scores) / len(valid_scores), 2) if valid_scores else 0,
            "num_flagged": len(removed),
            "score_distribution": dict(score_dist),
        },
        "nuggets": final,
        "removed": removed,
    }


def run_unified(config_path="config.yaml", reprocess=False, regenerate=False,
                review=False, paper_ids=None, dry_run=False, limit=None):
    """Run the unified nugget pipeline.

    Args:
        reprocess: If True, skip extraction and run quality+augment on existing
                   nuggets from nugget_dir. Useful for re-tuning thresholds.
        regenerate: If True, force re-extraction even if unified output exists.
                    Deletes old unified output before processing.
        review: If True, re-score and refine existing unified nuggets in-place.
                Reads from unified_dir, keeps good nuggets, improves weak ones,
                removes bad ones, gap-fills sparse chunks. Like reprocess but
                operates on unified output instead of raw extractions.
        paper_ids: Optional set of paper IDs to process. If None, process all.
        dry_run: If True, show what would be processed without actually running.
        limit: If set, process at most this many papers.
    """
    cfg = load_config(config_path)
    chunk_dir = cfg["paths"]["chunk_dir"]
    nugget_dir = cfg["paths"]["nugget_dir"]
    unified_dir = cfg["paths"].get("unified_dir", "corpus/nuggets_unified")
    ncfg = cfg.get("nuggets", {})
    ext_cfg = ncfg.get("extraction", {})
    qcfg = ncfg.get("quality", {})
    acfg = ncfg.get("augmentation", {})
    ucfg = ncfg.get("unified", {})
    max_workers = int(os.environ.get("MAX_NUM_SEQS", ucfg.get("max_workers", ext_cfg.get("max_workers", 8))))
    max_model_len = ncfg.get("vllm", {}).get("max_model_len", 8192)
    os.makedirs(unified_dir, exist_ok=True)

    clients, model = make_llm_clients(cfg)
    num_instances = len(clients)
    backend = ncfg.get("backend", "vllm")
    extra_body = {"chat_template_kwargs": {"enable_thinking": False}} if backend == "vllm" else None
    self_score = ucfg.get("self_score", False)
    print(f"[unified] vLLM instances: {num_instances}, self_score: {self_score}")

    # Resume: skip papers with existing unified output
    def _done(paper_id):
        if regenerate or review:
            return False  # force re-processing
        path = os.path.join(unified_dir, f"{paper_id}.jsonl")
        return os.path.exists(path)

    # Enumerate papers from chunk_dir (always needed)
    chunk_files = sorted(f for f in os.listdir(chunk_dir) if f.endswith(".json"))
    to_process = []
    skipped = 0
    for fname in chunk_files:
        paper_id = fname.replace(".json", "")
        # Filter by paper_ids if provided
        if paper_ids is not None and paper_id not in paper_ids:
            skipped += 1
            continue
        if _done(paper_id):
            skipped += 1
        elif reprocess or review:
            # In reprocess/review mode, need existing nuggets
            src_dir = unified_dir if review else nugget_dir
            nug_path = os.path.join(src_dir, f"{paper_id}.jsonl")
            if os.path.exists(nug_path):
                to_process.append(paper_id)
            else:
                skipped += 1  # no nuggets to reprocess/review
        else:
            to_process.append(paper_id)

    if regenerate:
        mode_label = "regenerate (re-extract+quality+augment)"
    elif review:
        mode_label = "review (score+improve+gap-fill unified nuggets)"
    elif reprocess:
        mode_label = "reprocess (quality+augment)"
    else:
        mode_label = "full (extract+quality+augment)"
    if limit and len(to_process) > limit:
        to_process = to_process[:limit]
        print(f"[unified] --limit {limit}: truncated to {len(to_process)} papers")

    print(f"[unified] {mode_label}: {len(to_process)} papers ({skipped} skipped) via {model}, workers={max_workers}")

    if dry_run:
        print(f"\n[unified] DRY RUN — would process {len(to_process)} papers:")
        for pid in to_process:
            chunk_path = os.path.join(chunk_dir, f"{pid}.json")
            n_chunks = 0
            try:
                data = load_json(chunk_path)
                n_chunks = len([c for c in data.get("chunks", []) if len(c.get("text", "").strip()) >= 50])
            except Exception:
                pass
            src = "chunks"
            if reprocess or review:
                src_dir = unified_dir if review else nugget_dir
                nug_path = os.path.join(src_dir, f"{pid}.jsonl")
                try:
                    nugs = [n for n in load_jsonl(nug_path) if not n.get("_removed")]
                    src = f"{len(nugs)} existing nuggets"
                except Exception:
                    src = "nuggets (missing)"
            print(f"  {pid}: {n_chunks} chunks, source={src}")
        print(f"\n[unified] DRY RUN complete. Use without --dry-run to execute.")
        return

    if not to_process:
        print("[unified] Nothing to process.")
        return

    # Load manifest for paper metadata (title, authors, year, venue)
    corpus_dir = cfg["paths"].get("corpus_dir", "corpus")
    manifest_path = os.path.join(corpus_dir, "manifest.json")
    manifest_by_id = {}
    if os.path.exists(manifest_path):
        manifest = load_json(manifest_path)
        for entry in manifest:
            pid = entry.get("paper_id", "")
            if pid:
                manifest_by_id[pid] = {
                    "title": entry.get("title", ""),
                    "authors": entry.get("authors", []),
                    "year": entry.get("year"),
                    "venue": entry.get("venue", ""),
                }
        print(f"[unified] Loaded manifest: {len(manifest_by_id)} papers with metadata")
    else:
        print(f"[unified] No manifest found at {manifest_path}, proceeding without paper metadata")

    # Load chunk data (needed for both modes)
    paper_chunks = {}
    paper_nuggets = {}  # only for reprocess mode
    load_failed = 0
    for paper_id in to_process:
        try:
            chunk_data = load_json(os.path.join(chunk_dir, f"{paper_id}.json"))
            chunks = [c for c in chunk_data.get("chunks", []) if len(c.get("text", "").strip()) >= 50]
            if not chunks and not reprocess:
                save_jsonl([], os.path.join(unified_dir, f"{paper_id}.jsonl"))
                continue
            if reprocess or review:
                src_dir = unified_dir if review else nugget_dir
                nugs = [n for n in load_jsonl(os.path.join(src_dir, f"{paper_id}.jsonl"))
                        if not n.get("_removed")]
                if nugs:
                    paper_nuggets[paper_id] = nugs
                else:
                    continue  # skip before adding to paper_chunks

            paper_chunks[paper_id] = chunks
        except Exception as e:
            print(f"  ERROR loading {paper_id}: {e}")
            load_failed += 1

    total_papers = len(paper_chunks)
    counters = {"done": 0, "total": total_papers, "nuggets": 0, "removed": 0}
    print_lock = threading.Lock()
    t_start = time.time()

    def _on_result(paper_id, result):
        save_jsonl(result["nuggets"], os.path.join(unified_dir, f"{paper_id}.jsonl"),
                   removed=result.get("removed"))
        with print_lock:
            counters["done"] += 1
            counters["nuggets"] += result["num_nuggets"]
            counters["removed"] += result["num_removed"]
            elapsed = time.time() - t_start
            rate = counters["done"] / elapsed * 3600 if elapsed > 0 else 0
            sys.stderr.write(
                f"\r\033[K  [{counters['done']}/{total_papers}] {paper_id[:40]:40s} "
                f"-> {result['num_nuggets']} nuggets "
                f"({result['num_improved']} improved, {result['num_gap_filled']} gap, "
                f"{result['num_removed']} removed) "
                f"[{rate:.0f} papers/hr]\n"
            )
            sys.stderr.flush()

    _rr_counter = itertools.count()
    _rr_lock = threading.Lock()

    def _worker(paper_id):
        with _rr_lock:
            client = clients[next(_rr_counter) % num_instances]
        meta = manifest_by_id.get(paper_id)
        if reprocess or review:
            result = _process_paper_reprocess(
                client, paper_id, paper_nuggets[paper_id],
                paper_chunks[paper_id], model, qcfg, acfg, ucfg,
                extra_body=extra_body, print_lock=print_lock, counters=counters,
                max_model_len=max_model_len,
            )
        else:
            result = _process_paper_unified(
                client, paper_id, paper_chunks[paper_id], model,
                ext_cfg, qcfg, acfg, ucfg,
                extra_body=extra_body, print_lock=print_lock, counters=counters,
                max_model_len=max_model_len, paper_meta=meta,
                self_score=self_score,
            )
        _on_result(paper_id, result)
        return result

    success = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_worker, pid): pid for pid in paper_chunks}
        for fut in as_completed(futures):
            pid = futures[fut]
            try:
                fut.result()
                success += 1
            except Exception as e:
                with print_lock:
                    sys.stderr.write(f"\r\033[K  ERROR {pid}: {e}\n")
                failed += 1

    elapsed = time.time() - t_start
    mins, secs = divmod(int(elapsed), 60)
    hrs, mins = divmod(mins, 60)
    sys.stderr.write("\r\033[K")
    print(
        f"\n[unified] Done in {hrs}h{mins:02d}m{secs:02d}s: "
        f"{success} papers, {counters['nuggets']} nuggets, "
        f"{counters['removed']} removed, {failed} failed, {skipped} skipped"
    )


def _select_oldest(unified_dir, chunk_dir, n):
    """Return the N paper IDs with the oldest unified output (by file mtime)."""
    entries = []
    for fname in os.listdir(unified_dir):
        if not fname.endswith(".jsonl"):
            continue
        paper_id = fname.replace(".jsonl", "")
        # Only consider papers that have chunks (still in corpus)
        if not os.path.exists(os.path.join(chunk_dir, f"{paper_id}.json")):
            continue
        path = os.path.join(unified_dir, fname)
        try:
            mtime = os.path.getmtime(path)
            entries.append((mtime, paper_id))
        except OSError:
            continue
    entries.sort()  # oldest first
    selected = [pid for _, pid in entries[:n]]
    if selected:
        from datetime import datetime
        oldest_ts = entries[0][0] if entries else 0
        newest_ts = entries[min(n, len(entries)) - 1][0] if entries else 0
        print(f"[unified] Selected {len(selected)} oldest papers "
              f"(generated {datetime.fromtimestamp(oldest_ts):%Y-%m-%d %H:%M} "
              f"to {datetime.fromtimestamp(newest_ts):%Y-%m-%d %H:%M})")
    return set(selected)


def main():
    ap = argparse.ArgumentParser(description="Unified nugget pipeline (extract+quality+augment)")
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("--reprocess", action="store_true",
                    help="Skip extraction; run quality+augment on existing nuggets from nugget_dir")
    ap.add_argument("--regenerate", action="store_true",
                    help="Force re-extraction with current prompt, even if unified output exists")
    ap.add_argument("--review", action="store_true",
                    help="Review existing unified nuggets: score, improve weak, remove bad, gap-fill")
    ap.add_argument("--paper-ids", type=str, default=None,
                    help="Comma-separated paper IDs to process (default: all)")
    ap.add_argument("--paper-ids-file", type=str, default=None,
                    help="File with one paper ID per line to process")
    ap.add_argument("--oldest", type=int, default=None, metavar="N",
                    help="Select the N papers with the oldest unified output for regeneration")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would be processed without actually running")
    ap.add_argument("--limit", type=int, default=None, metavar="N",
                    help="Process at most N papers (useful for testing)")
    args = ap.parse_args()

    paper_ids = None
    if args.oldest:
        if not args.regenerate and not args.review:
            args.review = True
            print("[unified] --oldest implies --review (use --regenerate to force re-extraction)")
        cfg = load_config(args.config)
        chunk_dir = cfg["paths"]["chunk_dir"]
        unified_dir = cfg["paths"].get("unified_dir", "corpus/nuggets_unified")
        paper_ids = _select_oldest(unified_dir, chunk_dir, args.oldest)
    elif args.paper_ids:
        paper_ids = set(p.strip() for p in args.paper_ids.split(",") if p.strip())
    elif args.paper_ids_file:
        with open(args.paper_ids_file) as f:
            paper_ids = set(line.strip() for line in f if line.strip() and not line.startswith("#"))
    if paper_ids:
        print(f"[unified] Filtering to {len(paper_ids)} paper(s)")

    run_unified(args.config, reprocess=args.reprocess, regenerate=args.regenerate,
                review=args.review, paper_ids=paper_ids,
                dry_run=args.dry_run, limit=args.limit)


if __name__ == "__main__":
    main()
