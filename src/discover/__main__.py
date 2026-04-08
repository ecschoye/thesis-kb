"""CLI entry point: python -m src.discover [options]"""

import argparse
import json
import sys


def cmd_run(args):
    """Run paper discovery."""
    from src.discover.discover import run_discovery

    download = not args.no_download
    download_top = args.download_top if args.download_top is not None else args.max_results

    report = run_discovery(
        config_path=args.config,
        bib_path=args.bib,
        max_results=args.max_results,
        date_from=args.date_from,
        sources=tuple(args.sources),
        use_embeddings=not args.no_embeddings,
        auto_ingest=download,
        ingest_top=download_top,
        citation_expand_n=args.citation_expand,
        local_papers_dir=args.local_papers,
    )

    if args.json:
        json.dump(report, sys.stdout, indent=2, ensure_ascii=False)
        print()


def cmd_ledger(args):
    """Query the persistent candidate ledger."""
    from src.discover.discover import _load_ledger, _get_corpus_dir
    import os

    corpus_dir = _get_corpus_dir(args.config)
    ledger_path = os.path.join(corpus_dir, "discovery_ledger.json")

    if not os.path.exists(ledger_path):
        print("No ledger found. Run a discovery first.")
        return

    ledger = _load_ledger(ledger_path)
    entries = ledger["entries"]
    stats = ledger["stats"]

    if args.json:
        json.dump(ledger, sys.stdout, indent=2, ensure_ascii=False)
        print()
        return

    # Filter
    items = list(entries.values())
    if args.status:
        items = [e for e in items if e.get("status") == args.status]
    if args.min_score:
        items = [e for e in items if e.get("best_score", 0) >= args.min_score]

    # Sort
    sort_key = args.sort
    if sort_key == "score":
        items.sort(key=lambda e: e.get("best_score", 0), reverse=True)
    elif sort_key == "citations":
        items.sort(key=lambda e: e.get("citation_count", 0), reverse=True)
    elif sort_key == "recent":
        items.sort(key=lambda e: e.get("first_seen", ""), reverse=True)
    elif sort_key == "seen":
        items.sort(key=lambda e: e.get("times_seen", 0), reverse=True)

    # Limit
    items = items[:args.limit]

    # Display
    print(f"\n{'='*60}")
    print(f"  Discovery Ledger — {len(entries)} total candidates")
    print(f"{'='*60}")
    print(f"  Runs: {stats.get('total_runs', 0)} | "
          f"First: {stats.get('first_run', '?')[:10]} | "
          f"Last: {stats.get('last_run', '?')[:10]}")

    status_counts = {}
    for e in entries.values():
        s = e.get("status", "candidate")
        status_counts[s] = status_counts.get(s, 0) + 1
    print(f"  Status: {', '.join(f'{k}: {v}' for k, v in sorted(status_counts.items()))}")

    if args.status:
        print(f"  Filter: status={args.status}")
    if args.min_score:
        print(f"  Filter: score >= {args.min_score}")

    print(f"\n  Showing {len(items)} entries (sorted by {sort_key}):")
    print(f"  {'-'*56}")

    for i, e in enumerate(items):
        score = e.get("best_score", 0)
        title = e.get("title", "?")[:70]
        year = e.get("year", "?")
        cites = e.get("citation_count", 0)
        seen = e.get("times_seen", 1)
        status = e.get("status", "candidate")
        authors = e.get("authors", [])
        author_str = authors[0].split()[-1] if authors else "?"
        if len(authors) > 1:
            author_str += " et al."
        arxiv = e.get("arxiv_id", "")

        status_icon = {"candidate": " ", "downloaded": "D", "rejected": "X", "ingested": "K"}
        icon = status_icon.get(status, "?")

        print(f"  {i+1:>4}. [{score:.2f}] [{icon}] {title}")
        print(f"        {author_str} ({year}) | cites: {cites} | seen: {seen}x"
              + (f" | arXiv:{arxiv}" if arxiv else ""))

    print()


def main():
    ap = argparse.ArgumentParser(
        description="Paper discovery tool for thesis-kb"
    )
    sub = ap.add_subparsers(dest="command")

    # --- run ---
    run_p = sub.add_parser("run", help="Search for new papers")
    run_p.add_argument("-c", "--config", default="config.yaml")
    run_p.add_argument("--bib", default=None)
    run_p.add_argument("-n", "--max-results", type=int, default=100)
    run_p.add_argument("--sources", nargs="+", default=["arxiv", "s2"],
                       choices=["arxiv", "s2", "openalex"])
    run_p.add_argument("--date-from", default=None)
    run_p.add_argument("--no-embeddings", action="store_true")
    run_p.add_argument("--no-download", action="store_true")
    run_p.add_argument("--download-top", type=int, default=None)
    run_p.add_argument("--citation-expand", type=int, default=20)
    run_p.add_argument("--local-papers", default=None)
    run_p.add_argument("--json", action="store_true")

    # --- ledger ---
    led_p = sub.add_parser("ledger", help="Browse the accumulated candidate ledger")
    led_p.add_argument("-c", "--config", default="config.yaml")
    led_p.add_argument("--status", choices=["candidate", "downloaded", "rejected", "ingested"],
                       help="Filter by status")
    led_p.add_argument("--min-score", type=float, help="Minimum relevance score")
    led_p.add_argument("--sort", default="score",
                       choices=["score", "citations", "recent", "seen"],
                       help="Sort order (default: score)")
    led_p.add_argument("--limit", type=int, default=50, help="Max entries to show (default: 50)")
    led_p.add_argument("--json", action="store_true", help="Dump full ledger as JSON")

    args = ap.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "ledger":
        cmd_ledger(args)
    else:
        # No subcommand = run (backwards compatible)
        # Re-parse with run defaults
        ap.set_defaults(command="run", config="config.yaml", bib=None,
                        max_results=100, sources=["arxiv", "s2"], date_from=None,
                        no_embeddings=False, no_download=False, download_top=None,
                        citation_expand=20, local_papers=None, json=False)
        args = ap.parse_args()
        cmd_run(args)


if __name__ == "__main__":
    main()
