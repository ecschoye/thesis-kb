"""Apply quality scoring output to SQLite + ChromaDB.

After running quality.py (which writes to corpus/nuggets_quality/),
this script reads those scores and updates:
  - SQLite nuggets.thesis_relevance
  - ChromaDB metadata thesis_relevance

Usage:
    python scripts/apply_quality_scores.py [-c config.yaml] [--dry-run]
"""
import argparse
import json
import os
import sqlite3

import chromadb

from src.utils import load_config


def apply_scores(config_path="config.yaml", dry_run=False):
    cfg = load_config(config_path)
    kb_dir = cfg["paths"]["kb_dir"]
    corpus_dir = cfg["paths"]["corpus_dir"]
    quality_dir = os.path.join(corpus_dir, "nuggets_quality")
    store_cfg = cfg.get("store", {})

    if not os.path.exists(quality_dir):
        print(f"Quality dir not found: {quality_dir}")
        print("Run quality scoring first: python -m src.nuggets.quality -c config.yaml")
        return

    # Collect all scores
    quality_files = sorted(f for f in os.listdir(quality_dir) if f.endswith(".json"))
    print(f"Found {len(quality_files)} quality score files")

    scores: dict[str, int] = {}  # nugget_id -> thesis_relevance
    for fname in quality_files:
        data = json.loads(open(os.path.join(quality_dir, fname)).read())
        for item in data.get("scores", data if isinstance(data, list) else []):
            nid = item.get("nugget_id", "")
            tr = item.get("thesis_relevance", 0)
            if nid and isinstance(tr, (int, float)):
                scores[nid] = int(tr)

    print(f"Loaded {len(scores)} nugget scores")

    if not scores:
        print("No scores found. Check quality file format.")
        return

    # Distribution
    dist = {}
    for v in scores.values():
        dist[v] = dist.get(v, 0) + 1
    print("Score distribution:")
    for k in sorted(dist):
        print(f"  thesis_relevance={k}: {dist[k]} nuggets")

    if dry_run:
        print("[DRY RUN] Would update SQLite + ChromaDB. Exiting.")
        return

    # Update SQLite
    db_name = store_cfg.get("sqlite", {}).get("db_name", "nuggets.db")
    db_path = os.path.join(kb_dir, db_name)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    batch = [(tr, nid) for nid, tr in scores.items()]
    c.executemany("UPDATE nuggets SET thesis_relevance=? WHERE nugget_id=?", batch)
    conn.commit()
    conn.close()
    print(f"SQLite: updated {len(batch)} nuggets")

    # Update ChromaDB
    chroma_path = os.path.join(kb_dir, "chromadb")
    collection_name = store_cfg.get("chromadb", {}).get("collection_name", "thesis_nuggets")
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_collection(collection_name)

    # ChromaDB update in batches (avoid memory issues)
    batch_size = 500
    nids = list(scores.keys())
    for start in range(0, len(nids), batch_size):
        end = min(start + batch_size, len(nids))
        batch_ids = nids[start:end]
        batch_metas = [{"thesis_relevance": scores[nid]} for nid in batch_ids]
        collection.update(ids=batch_ids, metadatas=batch_metas)
        if end % 5000 == 0 or end == len(nids):
            print(f"  ChromaDB: {end}/{len(nids)} updated")

    print(f"ChromaDB: updated {len(nids)} nuggets")
    print("\nDone. The thesis_relevance boost in api.py is now active.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Apply quality scores to KB")
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("--dry-run", action="store_true", help="Show stats only, don't update")
    args = ap.parse_args()
    apply_scores(args.config, dry_run=args.dry_run)
