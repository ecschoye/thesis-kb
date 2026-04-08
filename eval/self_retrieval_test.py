#!/usr/bin/env python3
"""Self-retrieval test: use nugget questions as queries, measure recall@k and MRR.

Each nugget's `question` field is used as a query. The ground truth is that the
nugget itself should appear in the top-k results. This tests embedding quality
and vector search without requiring human-labeled relevance pairs.

Usage:
    python eval/self_retrieval_test.py                    # 500 samples, default
    python eval/self_retrieval_test.py --n 100 --k 10     # 100 samples, top-10
    python eval/self_retrieval_test.py --full              # all nuggets (slow)
    python eval/self_retrieval_test.py --type method       # only method nuggets
    python eval/self_retrieval_test.py --paper 2401_12345  # specific paper
"""
import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def load_nuggets_from_sqlite(kb, type_filter=None, paper_filter=None):
    """Load nuggets with their IDs from SQLite."""
    query = "SELECT nugget_id, paper_id, question, type, section, overall_score FROM nuggets"
    conditions = []
    params = []
    if type_filter:
        conditions.append("type = ?")
        params.append(type_filter)
    if paper_filter:
        conditions.append("paper_id LIKE ?")
        params.append(f"%{paper_filter}%")
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    rows = kb.db.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def sample_nuggets(nuggets, n, stratify_by="type"):
    """Stratified sample of nuggets."""
    if n >= len(nuggets):
        return nuggets

    groups = defaultdict(list)
    for nug in nuggets:
        key = nug.get(stratify_by, "unknown") or "unknown"
        groups[key] = groups.get(key, [])
        groups[key].append(nug)

    # Proportional allocation per group
    sampled = []
    for key, group in groups.items():
        group_n = max(1, round(n * len(group) / len(nuggets)))
        sampled.extend(random.sample(group, min(group_n, len(group))))

    # Trim or pad to exactly n
    if len(sampled) > n:
        sampled = random.sample(sampled, n)
    elif len(sampled) < n:
        remaining = [nug for nug in nuggets if nug not in sampled]
        sampled.extend(random.sample(remaining, min(n - len(sampled), len(remaining))))

    return sampled


def run_self_retrieval(nuggets, kb, embed_client, embed_model, instruction, k=20,
                       verbose=False, progress_interval=50):
    """Run self-retrieval test and return per-nugget results."""
    from src.embed.embedder import embed_batch

    results = []
    t0 = time.time()

    for i, nug in enumerate(nuggets):
        nid = nug["nugget_id"]
        question = nug["question"]

        # Embed the question
        query_text = f"Instruct: {instruction}\nQuery: {question}" if instruction else question
        try:
            emb = embed_batch(embed_client, [query_text], embed_model)
        except Exception as e:
            results.append({
                "nugget_id": nid, "paper_id": nug["paper_id"],
                "type": nug["type"], "section": nug.get("section", ""),
                "rank": None, "found": False, "error": str(e),
            })
            continue

        # Query ChromaDB
        search_results = kb.collection.query(
            query_embeddings=emb,
            n_results=k,
            include=["metadatas"],
        )

        # Find rank of source nugget
        retrieved_ids = search_results["ids"][0]
        rank = None
        for ri, rid in enumerate(retrieved_ids):
            if rid == nid:
                rank = ri + 1  # 1-indexed
                break

        results.append({
            "nugget_id": nid,
            "paper_id": nug["paper_id"],
            "type": nug["type"],
            "section": nug.get("section", ""),
            "overall_score": nug.get("overall_score"),
            "rank": rank,
            "found": rank is not None,
        })

        if progress_interval and (i + 1) % progress_interval == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(nuggets) - i - 1) / rate if rate > 0 else 0
            recall = sum(1 for r in results if r["found"]) / len(results)
            sys.stderr.write(
                f"\r\033[K  [{i+1}/{len(nuggets)}] recall@{k}={recall:.3f} "
                f"({rate:.1f} queries/s, ETA {eta:.0f}s)"
            )
            sys.stderr.flush()

    if progress_interval:
        sys.stderr.write("\r\033[K")
        sys.stderr.flush()

    return results


def compute_metrics(results, k):
    """Compute aggregate metrics from per-nugget results."""
    n = len(results)
    if n == 0:
        return {}

    found = [r for r in results if r["found"]]
    ranks = [r["rank"] for r in found]

    recall_at_k = len(found) / n
    mrr = sum(1.0 / r for r in ranks) / n if ranks else 0.0

    # Recall at various k values
    recall_at = {}
    for cutoff in [1, 3, 5, 10, 20, 50]:
        if cutoff <= k:
            recall_at[cutoff] = sum(1 for r in ranks if r <= cutoff) / n

    # Rank distribution
    rank_dist = defaultdict(int)
    for r in ranks:
        if r <= 1:
            rank_dist["rank_1"] += 1
        elif r <= 5:
            rank_dist["rank_2-5"] += 1
        elif r <= 10:
            rank_dist["rank_6-10"] += 1
        elif r <= 20:
            rank_dist["rank_11-20"] += 1
        else:
            rank_dist["rank_21+"] += 1
    rank_dist["not_found"] = n - len(found)

    return {
        "n": n,
        "recall_at_k": round(recall_at_k, 4),
        "mrr": round(mrr, 4),
        "recall_at": {f"@{c}": round(v, 4) for c, v in sorted(recall_at.items())},
        "rank_distribution": dict(rank_dist),
        "mean_rank_when_found": round(sum(ranks) / len(ranks), 2) if ranks else None,
        "median_rank_when_found": sorted(ranks)[len(ranks) // 2] if ranks else None,
    }


def compute_breakdowns(results, k):
    """Compute metrics broken down by type and section."""
    breakdowns = {}

    for key_field in ("type", "section"):
        groups = defaultdict(list)
        for r in results:
            groups[r.get(key_field, "unknown") or "unknown"].append(r)

        breakdown = {}
        for group_name, group_results in sorted(groups.items()):
            if len(group_results) < 3:
                continue  # skip tiny groups
            m = compute_metrics(group_results, k)
            breakdown[group_name] = {
                "n": m["n"],
                "recall": m["recall_at_k"],
                "mrr": m["mrr"],
            }
        breakdowns[key_field] = breakdown

    return breakdowns


def main():
    ap = argparse.ArgumentParser(description="Self-retrieval test: measure recall@k and MRR")
    ap.add_argument("--n", type=int, default=500, help="Number of nuggets to sample (default: 500)")
    ap.add_argument("--k", type=int, default=20, help="Top-k to check (default: 20)")
    ap.add_argument("--full", action="store_true", help="Test all nuggets (overrides --n)")
    ap.add_argument("--type", default=None, help="Filter to specific nugget type")
    ap.add_argument("--paper", default=None, help="Filter to specific paper (substring match)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    ap.add_argument("--config", "-c", default=None, help="Config file")
    ap.add_argument("--verbose", action="store_true", help="Show per-nugget results")
    ap.add_argument("--output", "-o", default=None, help="Save results as JSON")
    args = ap.parse_args()

    random.seed(args.seed)

    config_path = args.config or os.environ.get("THESIS_KB_CONFIG", "config-ollama.yaml")
    from src.utils import load_config
    cfg = load_config(config_path)
    from src.query import ThesisKB
    kb = ThesisKB(config_path)
    from src.embed.embedder import make_embed_client
    embed_client, embed_model = make_embed_client(cfg)
    instruction = cfg.get("embed", {}).get("embedding", {}).get("query_instruction", "")

    print(f"KB: {kb.collection.count()} nuggets in ChromaDB")

    # Load and sample nuggets
    all_nuggets = load_nuggets_from_sqlite(kb, type_filter=args.type, paper_filter=args.paper)
    print(f"Eligible nuggets: {len(all_nuggets)}")

    if args.full:
        sample = all_nuggets
    else:
        sample = sample_nuggets(all_nuggets, args.n)

    type_dist = defaultdict(int)
    for nug in sample:
        type_dist[nug["type"]] += 1
    print(f"Sample: {len(sample)} nuggets, k={args.k}")
    print(f"  Type distribution: {dict(sorted(type_dist.items()))}")
    print()

    # Run self-retrieval
    t0 = time.time()
    results = run_self_retrieval(
        sample, kb, embed_client, embed_model, instruction,
        k=args.k, verbose=args.verbose,
    )
    elapsed = time.time() - t0

    # Compute metrics
    metrics = compute_metrics(results, args.k)
    breakdowns = compute_breakdowns(results, args.k)

    # Display results
    print(f"{'=' * 60}")
    print(f"Self-Retrieval Results (n={metrics['n']}, k={args.k}, {elapsed:.1f}s)")
    print(f"{'=' * 60}")
    print(f"  MRR:        {metrics['mrr']:.4f}")
    print(f"  Recall@{args.k}:  {metrics['recall_at_k']:.4f}")
    for cutoff, val in sorted(metrics.get("recall_at", {}).items()):
        print(f"  Recall{cutoff}: {val:.4f}")
    print(f"  Mean rank (when found):   {metrics.get('mean_rank_when_found', 'N/A')}")
    print(f"  Median rank (when found): {metrics.get('median_rank_when_found', 'N/A')}")
    print()

    print("Rank distribution:")
    for bucket, count in sorted(metrics.get("rank_distribution", {}).items()):
        pct = count / metrics["n"] * 100
        bar = "#" * int(pct / 2)
        print(f"  {bucket:12s}: {count:4d} ({pct:5.1f}%) {bar}")
    print()

    # Breakdowns
    for field, breakdown in breakdowns.items():
        print(f"By {field}:")
        print(f"  {'Group':20s} {'N':>5s} {'Recall':>8s} {'MRR':>8s}")
        print(f"  {'-' * 43}")
        for group, m in sorted(breakdown.items(), key=lambda x: -x[1]["recall"]):
            print(f"  {group:20s} {m['n']:5d} {m['recall']:8.4f} {m['mrr']:8.4f}")
        print()

    # Verbose: show failures
    if args.verbose:
        failures = [r for r in results if not r["found"]]
        if failures:
            print(f"Failed retrievals ({len(failures)}):")
            for r in failures[:20]:
                nug = next((n for n in sample if n["nugget_id"] == r["nugget_id"]), {})
                print(f"  [{r['type']}] {r['nugget_id'][:50]}")
                print(f"    Q: {nug.get('question', '?')[:80]}")

    # Save output
    if args.output:
        output = {
            "config": {
                "n": len(sample), "k": args.k, "seed": args.seed,
                "type_filter": args.type, "paper_filter": args.paper,
                "config_path": config_path,
            },
            "metrics": metrics,
            "breakdowns": breakdowns,
            "elapsed_s": round(elapsed, 2),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "per_nugget": results,
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {args.output}")

    # Exit code: 0 if recall@k > 0.5, 1 otherwise (rough health check)
    sys.exit(0 if metrics["recall_at_k"] > 0.5 else 1)


if __name__ == "__main__":
    main()
