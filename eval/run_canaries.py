#!/usr/bin/env python3
"""Run canary queries against the KB and check for expected papers in top-k.

Usage:
    python eval/run_canaries.py                          # run all modes
    python eval/run_canaries.py --mode background check  # specific modes
    python eval/run_canaries.py --k 10                   # top-10 instead of default
    python eval/run_canaries.py --verbose                 # show retrieved paper_ids
    python eval/run_canaries.py --api http://localhost:8001  # use running API server
"""
import argparse
import json
import os
import sys
import time

# Ensure project root is on sys.path when run as a script
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def load_canaries(path="eval/canary_queries.json"):
    with open(path) as f:
        data = json.load(f)
    # Strip _meta key
    return {k: v for k, v in data.items() if not k.startswith("_")}


def run_canary_offline(query, mode, k, kb, embed_client, embed_model, embed_instruction):
    """Run a canary query using direct KB access (no API server needed)."""
    from src.embed.embedder import embed_batch, format_nugget_text

    # Embed the query
    query_text = f"Instruct: {embed_instruction}\nQuery: {query}" if embed_instruction else query
    embeddings = embed_batch(embed_client, [query_text], embed_model)

    # Vector search
    results = kb.collection.query(
        query_embeddings=embeddings,
        n_results=k,
        include=["metadatas", "documents", "distances"],
    )

    # Extract paper_ids from results
    paper_ids = set()
    for meta in results["metadatas"][0]:
        pid = meta.get("paper_id", "")
        if pid:
            paper_ids.add(pid)

    return paper_ids


def run_canary_api(query, mode, k, api_url):
    """Run a canary query via the running API server."""
    import urllib.request
    import urllib.parse

    url = f"{api_url}/api/chat"
    payload = json.dumps({
        "messages": [{"role": "user", "content": query}],
        "mode": mode,
        "stream": False,
    }).encode()

    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            # SSE response — parse event lines for nugget data
            paper_ids = set()
            for line in resp.read().decode().split("\n"):
                if line.startswith("data: "):
                    try:
                        event = json.loads(line[6:])
                        if event.get("type") == "nuggets":
                            for n in event.get("nuggets", []):
                                pid = n.get("paper_id", "")
                                if pid:
                                    paper_ids.add(pid)
                    except json.JSONDecodeError:
                        pass
            return paper_ids
    except Exception as e:
        print(f"  API error: {e}", file=sys.stderr)
        return set()


def run_canary_direct(query, mode, k, kb):
    """Run a canary query using direct SQLite BM25 (no embedding needed)."""
    if not kb or not kb.db:
        return set()

    # Escape FTS5 query: quote each word to avoid syntax errors with special chars
    import re
    words = re.findall(r'\w+', query)
    fts_query = " ".join(f'"{w}"' for w in words if len(w) > 2)
    if not fts_query:
        return set()

    # BM25 search via FTS5
    rows = kb.db.execute(
        "SELECT n.paper_id FROM nuggets_fts f "
        "JOIN nuggets n ON f.rowid = n.rowid "
        "WHERE nuggets_fts MATCH ? "
        "ORDER BY rank LIMIT ?",
        (fts_query, k),
    ).fetchall()

    paper_ids = set()
    for r in rows:
        pid = r[0] if isinstance(r, (tuple, list)) else r["paper_id"]
        if pid:
            paper_ids.add(pid)
    return paper_ids


def main():
    ap = argparse.ArgumentParser(description="Run canary queries for retrieval regression testing")
    ap.add_argument("--fixture", default="eval/canary_queries.json", help="Path to canary fixture")
    ap.add_argument("--mode", nargs="*", help="Modes to test (default: all)")
    ap.add_argument("--k", type=int, default=20, help="Top-k to check (default: 20)")
    ap.add_argument("--verbose", action="store_true", help="Show retrieved paper_ids")
    ap.add_argument("--config", "-c", default=None, help="Config file (default: from THESIS_KB_CONFIG or config-ollama.yaml)")
    ap.add_argument("--api", default=None, help="API URL (e.g. http://localhost:8001) — uses running server instead of offline")
    ap.add_argument("--bm25-only", action="store_true", help="Use BM25 search only (no embedding needed)")
    args = ap.parse_args()

    canaries = load_canaries(args.fixture)
    modes_to_test = args.mode or list(canaries.keys())

    # Count queries with expected_paper_ids
    total_queries = 0
    testable_queries = 0
    for mode in modes_to_test:
        for q in canaries.get(mode, []):
            total_queries += 1
            if q.get("expected_paper_ids"):
                testable_queries += 1

    print(f"Canary test: {total_queries} queries across {len(modes_to_test)} modes "
          f"({testable_queries} with expected papers, {total_queries - testable_queries} uncurated)")

    if testable_queries == 0:
        print("\nNo queries have expected_paper_ids set. Curate the fixture first.")
        print("Run queries in the web UI and add paper_ids to eval/canary_queries.json")
        return

    # Initialize KB access
    kb = None
    embed_client = None
    embed_model = None
    embed_instruction = None

    if args.api:
        print(f"Using API server at {args.api}")
    else:
        config_path = args.config or os.environ.get("THESIS_KB_CONFIG", "config-ollama.yaml")
        from src.utils import load_config
        cfg = load_config(config_path)
        from src.query import ThesisKB
        kb = ThesisKB(config_path)
        kb_dir = cfg["paths"]["kb_dir"]
        print(f"Loaded KB from {kb_dir}: {kb.collection.count() if kb.collection else 0} nuggets")

        if not args.bm25_only:
            from src.embed.embedder import make_embed_client
            embed_client, embed_model = make_embed_client(cfg)
            embed_instruction = cfg.get("embed", {}).get("embedding", {}).get("query_instruction", "")

    # Run canaries
    passed = 0
    failed = 0
    skipped = 0
    failures = []

    for mode in modes_to_test:
        queries = canaries.get(mode, [])
        if not queries:
            continue
        print(f"\n{'=' * 60}")
        print(f"Mode: {mode} ({len(queries)} queries)")
        print(f"{'=' * 60}")

        for qi, q in enumerate(queries):
            query_text = q["query"]
            expected = set(q.get("expected_paper_ids", []))

            if not expected:
                if args.verbose:
                    print(f"  [{qi+1}] SKIP (no expected papers): {query_text[:60]}")
                skipped += 1
                continue

            # Run the query
            t0 = time.time()
            if args.api:
                retrieved_papers = run_canary_api(query_text, mode, args.k, args.api)
            elif args.bm25_only:
                retrieved_papers = run_canary_direct(query_text, mode, args.k, kb)
            else:
                retrieved_papers = run_canary_offline(query_text, mode, args.k, kb, embed_client, embed_model, embed_instruction)
            elapsed = time.time() - t0

            # Check assertions
            found = expected & retrieved_papers
            missing = expected - retrieved_papers

            if missing:
                status = "FAIL"
                failed += 1
                failures.append({
                    "mode": mode,
                    "query": query_text,
                    "missing": list(missing),
                    "found": list(found),
                })
            else:
                status = "PASS"
                passed += 1

            print(f"  [{qi+1}] {status} ({elapsed:.2f}s): {query_text[:60]}")
            if missing:
                for pid in missing:
                    print(f"       MISSING: {pid[:60]}")
            if args.verbose and retrieved_papers:
                print(f"       Retrieved {len(retrieved_papers)} papers: {', '.join(list(retrieved_papers)[:5])}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"RESULTS: {passed} passed, {failed} failed, {skipped} skipped (uncurated)")
    print(f"{'=' * 60}")

    if failures:
        print(f"\nFailed canaries:")
        for f in failures:
            print(f"  [{f['mode']}] {f['query'][:50]}")
            for pid in f["missing"]:
                print(f"    missing: {pid[:60]}")

    # Exit code: 0 if all testable queries pass, 1 if any fail
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
