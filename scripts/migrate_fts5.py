"""Add FTS5 full-text search index to existing nuggets.db."""
import sqlite3
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_config


def migrate(config_path="config.yaml"):
    cfg = load_config(config_path)
    kb_dir = cfg["paths"]["kb_dir"]
    db_name = cfg.get("store", {}).get("sqlite", {}).get("db_name", "nuggets.db")
    db_path = os.path.join(kb_dir, db_name)

    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    c = conn.cursor()

    # Check if FTS5 table already exists
    tables = c.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='nuggets_fts'"
    ).fetchone()
    if tables:
        print("FTS5 table already exists. Dropping and rebuilding...")
        c.execute("DROP TABLE nuggets_fts")

    print("Creating FTS5 virtual table...")
    c.execute("""
        CREATE VIRTUAL TABLE nuggets_fts USING fts5(
            nugget_id UNINDEXED,
            question,
            answer,
            content='nuggets',
            content_rowid='rowid'
        )
    """)

    print("Populating FTS5 index...")
    c.execute("""
        INSERT INTO nuggets_fts(nugget_id, question, answer)
        SELECT nugget_id, question, answer FROM nuggets
    """)

    conn.commit()
    count = c.execute("SELECT COUNT(*) FROM nuggets_fts").fetchone()[0]
    conn.close()
    print(f"FTS5 index built: {count} entries")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="config-ollama.yaml")
    args = ap.parse_args()
    migrate(args.config)
