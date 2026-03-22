"""Unified nugget pipeline: extract → quality → augment in one per-paper pass.

Keeps chunks in memory, runs all three stages sequentially per paper,
saves one output JSON per paper with inlined quality scores and audit trail.

Usage:
    python -m src.nuggets.unified -c config.yaml              # full pipeline
    python -m src.nuggets.unified -c config.yaml --reprocess   # quality+augment only
"""
import os, json, sys, time, argparse, threading, itertools
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils import load_config, load_json, save_json, make_llm_client, make_llm_clients

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
    max_model_len=4096,
):
    """Full extract→quality→augment pipeline for one paper, all in memory.

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

    # ── Step 1: Extract nuggets from chunks ──────────────────────────────
    temp = ext_cfg.get("temperature", 0.1)
    max_tok = ext_cfg.get("max_tokens", 3000)
    max_retries = ext_cfg.get("max_retries", 3)
    retry_delay = ext_cfg.get("retry_base_delay", 2.0)

    all_raw_nuggets = []
    prior_questions = []
    chunk_by_id = {}

    for ci, c in enumerate(chunks):
        chunk_by_id[c["chunk_id"]] = c
        try:
            nuggets = _process_chunk(
                client, c, model, temp, max_tok,
                max_retries, retry_delay, paper_id, extra_body,
                prior_questions=prior_questions if prior_questions else None,
                max_model_len=max_model_len,
            )
        except Exception as e:
            _log(f"  WARN {short_id} chunk {ci}: {e}")
            nuggets = []
        all_raw_nuggets.extend(nuggets)
        prior_questions.extend(n["question"] for n in nuggets)

    # ── Step 2: Deduplicate ──────────────────────────────────────────────
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
    paper_title = _get_paper_title(deduped, paper_id)
    batch_size = qcfg.get("batch_size", 5)
    quality_by_id = {}

    for i in range(0, len(deduped), batch_size):
        batch = deduped[i:i + batch_size]
        results, err = rate_nugget_batch(
            client, batch, model, paper_title, paper_id, qcfg,
            max_model_len=max_model_len,
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

        new_nuggets = gapfill_chunk(client, chunk["text"], existing, model, acfg)
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
    valid_scores = [n["quality"]["overall"] for n in deduped if n["quality"]["overall"] > 0]
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
    max_model_len=4096,
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
            max_model_len=max_model_len,
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
        new_nuggets = gapfill_chunk(client, chunk["text"], existing, model, acfg)
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

    valid_scores = [n["quality"]["overall"] for n in nuggets if n["quality"]["overall"] > 0]
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


def run_unified(config_path="config.yaml", reprocess=False):
    """Run the unified nugget pipeline.

    Args:
        reprocess: If True, skip extraction and run quality+augment on existing
                   nuggets from nugget_dir. Useful for re-tuning thresholds.
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
    max_model_len = ncfg.get("vllm", {}).get("max_model_len", 4096)
    os.makedirs(unified_dir, exist_ok=True)

    clients, model = make_llm_clients(cfg)
    num_instances = len(clients)
    backend = ncfg.get("backend", "vllm")
    extra_body = {"chat_template_kwargs": {"enable_thinking": False}} if backend == "vllm" else None
    print(f"[unified] vLLM instances: {num_instances}")

    # Resume: skip papers with existing unified output
    def _done(paper_id):
        path = os.path.join(unified_dir, f"{paper_id}.json")
        if not os.path.exists(path):
            return False
        try:
            data = json.loads(open(path).read())
            return data.get("num_nuggets", 0) > 0 or len(data.get("removed", [])) > 0
        except (json.JSONDecodeError, OSError):
            return False

    # Enumerate papers from chunk_dir (always needed)
    chunk_files = sorted(f for f in os.listdir(chunk_dir) if f.endswith(".json"))
    to_process = []
    skipped = 0
    for fname in chunk_files:
        paper_id = fname.replace(".json", "")
        if _done(paper_id):
            skipped += 1
        elif reprocess:
            # In reprocess mode, need existing nuggets
            nug_path = os.path.join(nugget_dir, f"{paper_id}.json")
            if os.path.exists(nug_path):
                to_process.append(paper_id)
            else:
                skipped += 1  # no nuggets to reprocess
        else:
            to_process.append(paper_id)

    mode_label = "reprocess (quality+augment)" if reprocess else "full (extract+quality+augment)"
    print(f"[unified] {mode_label}: {len(to_process)} papers ({skipped} skipped) via {model}, workers={max_workers}")

    if not to_process:
        print("[unified] Nothing to process.")
        return

    # Load chunk data (needed for both modes)
    paper_chunks = {}
    paper_nuggets = {}  # only for reprocess mode
    load_failed = 0
    for paper_id in to_process:
        try:
            chunk_data = load_json(os.path.join(chunk_dir, f"{paper_id}.json"))
            chunks = [c for c in chunk_data.get("chunks", []) if len(c.get("text", "").strip()) >= 50]
            if not chunks and not reprocess:
                save_json(
                    {"paper_id": paper_id, "num_nuggets": 0, "num_removed": 0,
                     "num_improved": 0, "num_gap_filled": 0,
                     "quality_summary": {}, "nuggets": [], "removed": []},
                    os.path.join(unified_dir, f"{paper_id}.json"),
                )
                continue
            if reprocess:
                nug_data = load_json(os.path.join(nugget_dir, f"{paper_id}.json"))
                nugs = nug_data.get("nuggets", [])
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
        save_json(result, os.path.join(unified_dir, f"{paper_id}.json"))
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
        if reprocess:
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
                max_model_len=max_model_len,
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


def main():
    ap = argparse.ArgumentParser(description="Unified nugget pipeline (extract+quality+augment)")
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("--reprocess", action="store_true",
                    help="Skip extraction; run quality+augment on existing nuggets")
    args = ap.parse_args()
    run_unified(args.config, reprocess=args.reprocess)


if __name__ == "__main__":
    main()
