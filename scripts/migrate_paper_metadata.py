"""Migrate existing SQLite DB to include paper authority metadata from S2.

Adds citation_count, influential_citation_count, paper_type columns and
populates them by fetching from Semantic Scholar API.

Usage:
    python scripts/migrate_paper_metadata.py [-c config.yaml] [--dry-run]
"""
import argparse
import json
import os
import sqlite3
import time

from src.acquire.enrich import enrich_via_s2
from src.utils import load_config


def migrate(config_path="config.yaml", dry_run=False):
    cfg = load_config(config_path)
    kb_dir = cfg["paths"]["kb_dir"]
    db_name = cfg.get("store", {}).get("sqlite", {}).get("db_name", "nuggets.db")
    db_path = os.path.join(kb_dir, db_name)

    if not os.path.exists(db_path):
        print(f"DB not found: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # Add columns if they don't exist
    existing_cols = {row[1] for row in c.execute("PRAGMA table_info(papers)").fetchall()}
    for col, typedef in [
        ("citation_count", "INTEGER DEFAULT 0"),
        ("influential_citation_count", "INTEGER DEFAULT 0"),
        ("paper_type", "TEXT DEFAULT ''"),
    ]:
        if col not in existing_cols:
            print(f"  Adding column: {col}")
            c.execute(f"ALTER TABLE papers ADD COLUMN {col} {typedef}")
    conn.commit()

    # Load progress file (resume-safe)
    progress_path = os.path.join(kb_dir, ".authority_migration_progress.json")
    done_ids = set()
    if os.path.exists(progress_path):
        done_ids = set(json.loads(open(progress_path).read()))
        print(f"  Resuming: {len(done_ids)} papers already processed")

    # Fetch all papers
    papers = [dict(r) for r in c.execute("SELECT * FROM papers").fetchall()]
    print(f"  {len(papers)} papers total, {len(papers) - len(done_ids)} to process")

    if dry_run:
        print("  [DRY RUN] Would fetch S2 metadata and update DB. Exiting.")
        conn.close()
        return

    updated = 0
    failed = 0
    for i, paper in enumerate(papers):
        pid = paper["paper_id"]
        if pid in done_ids:
            continue

        # Already has authority data from a previous partial run
        if paper.get("citation_count") and paper["citation_count"] > 0:
            done_ids.add(pid)
            continue

        title = paper.get("title", "")
        arxiv_id = paper.get("arxiv_id")
        result = enrich_via_s2(title, arxiv_id=arxiv_id)

        if result:
            cc = result.get("citation_count", 0)
            icc = result.get("influential_citation_count", 0)
            pub_types = result.get("publication_types", [])
            paper_type = ",".join(pub_types) if isinstance(pub_types, list) else ""
            c.execute(
                "UPDATE papers SET citation_count=?, influential_citation_count=?, paper_type=? WHERE paper_id=?",
                (cc, icc, paper_type, pid),
            )
            updated += 1
        else:
            failed += 1

        done_ids.add(pid)

        # Save progress every 10 papers
        if (i + 1) % 10 == 0:
            conn.commit()
            with open(progress_path, "w") as f:
                json.dump(list(done_ids), f)
            processed = len(done_ids)
            print(f"  [{processed}/{len(papers)}] updated={updated} failed={failed}")

        time.sleep(1.0)  # S2 rate limit

    conn.commit()
    # Clean up progress file
    if os.path.exists(progress_path):
        os.remove(progress_path)

    print(f"\nMigration complete: {updated} updated, {failed} failed out of {len(papers)} papers")

    # Print summary stats
    rows = c.execute(
        "SELECT paper_type, COUNT(*), AVG(citation_count) FROM papers GROUP BY paper_type ORDER BY COUNT(*) DESC"
    ).fetchall()
    print("\nPaper type distribution:")
    for row in rows:
        ptype = row[0] or "(none)"
        print(f"  {ptype:30s}  count={row[1]:4d}  avg_citations={row[2]:.0f}")

    conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Migrate paper authority metadata from S2")
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("--dry-run", action="store_true", help="Check schema only, don't fetch S2 data")
    args = ap.parse_args()
    migrate(args.config, dry_run=args.dry_run)
