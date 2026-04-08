"""Build ChromaDB + SQLite knowledge base from embedded nuggets."""
import os
import json
import sqlite3
import argparse
import numpy as np
import chromadb
from src.utils import load_config, load_json, load_jsonl


def build_chromadb(nuggets, embeddings, kb_dir, collection_name, distance_fn="cosine"):
    """Insert nuggets and embeddings into ChromaDB."""
    chroma_path = os.path.join(kb_dir, "chromadb")
    os.makedirs(chroma_path, exist_ok=True)
    client = chromadb.PersistentClient(path=chroma_path)

    # Delete existing collection if present
    try:
        client.delete_collection(collection_name)
    except ValueError:
        pass  # collection doesn't exist

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": distance_fn},
    )

    # Batch insert (ChromaDB limit ~5000 per call)
    batch_size = 500
    for start in range(0, len(nuggets), batch_size):
        end = min(start + batch_size, len(nuggets))
        batch_nuggets = nuggets[start:end]
        batch_embs = embeddings[start:end].tolist()

        ids = [n["nugget_id"] for n in batch_nuggets]
        documents = [
            f"[{n.get('type', '')}] [{n.get('section', '')}] Q: {n['question']} A: {n['answer']}"
            for n in batch_nuggets
        ]
        metadatas = [
            {
                "paper_id": n.get("paper_id", ""),
                "type": n.get("type", ""),
                "confidence": n.get("confidence", ""),
                "section": n.get("section", ""),
                "thesis_relevance": n.get("thesis_relevance", 3),
                "source_file": n.get("source_file", ""),
            }
            for n in batch_nuggets
        ]

        collection.add(
            ids=ids,
            embeddings=batch_embs,
            documents=documents,
            metadatas=metadatas,
        )
        if end % 2000 == 0 or end == len(nuggets):
            print(f"  ChromaDB: {end}/{len(nuggets)} inserted")

    print(f"  ChromaDB collection '{collection_name}': {collection.count()} nuggets")
    return collection


def build_sqlite(nuggets, manifest, kb_dir, db_name="nuggets.db", cfg=None):
    """Build SQLite metadata database."""
    db_path = os.path.join(kb_dir, db_name)
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Create tables
    c.execute("""
        CREATE TABLE papers (
            paper_id TEXT PRIMARY KEY,
            title TEXT,
            authors TEXT,
            year INTEGER,
            arxiv_id TEXT,
            doi TEXT,
            abstract TEXT,
            source TEXT,
            citation_count INTEGER DEFAULT 0,
            influential_citation_count INTEGER DEFAULT 0,
            paper_type TEXT DEFAULT ''
        )
    """)

    c.execute("""
        CREATE TABLE nuggets (
            nugget_id TEXT PRIMARY KEY,
            paper_id TEXT,
            question TEXT,
            answer TEXT,
            type TEXT,
            confidence TEXT,
            section TEXT,
            source_chunk INTEGER,
            thesis_relevance INTEGER DEFAULT 0,
            overall_score REAL DEFAULT NULL,
            flagged INTEGER DEFAULT 0,
            source_file TEXT DEFAULT NULL,
            FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
        )
    """)

    c.execute("CREATE INDEX idx_nuggets_paper ON nuggets(paper_id)")
    c.execute("CREATE INDEX idx_nuggets_type ON nuggets(type)")
    c.execute("CREATE INDEX idx_nuggets_relevance ON nuggets(thesis_relevance)")
    c.execute("CREATE INDEX idx_nuggets_section ON nuggets(section)")
    c.execute("CREATE INDEX idx_nuggets_paper_type ON nuggets(paper_id, type)")
    c.execute("CREATE INDEX idx_nuggets_flagged ON nuggets(flagged)")
    c.execute("CREATE INDEX idx_nuggets_source_file ON nuggets(source_file)")

    # Code-paper cross-reference table
    c.execute("""
        CREATE TABLE IF NOT EXISTS code_links (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id TEXT NOT NULL,
            source_file TEXT NOT NULL,
            link_type TEXT NOT NULL,
            description TEXT,
            FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
        )
    """)
    c.execute("CREATE INDEX idx_code_links_paper ON code_links(paper_id)")
    c.execute("CREATE INDEX idx_code_links_file ON code_links(source_file)")

    # FTS5 full-text search index for BM25 retrieval
    c.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS nuggets_fts USING fts5(
            nugget_id UNINDEXED,
            question,
            answer,
            content='nuggets',
            content_rowid='rowid'
        )
    """)

    # Insert papers
    for paper in manifest:
        pub_types = paper.get("publication_types", [])
        paper_type = ",".join(pub_types) if isinstance(pub_types, list) else str(pub_types or "")
        c.execute(
            "INSERT OR IGNORE INTO papers VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                paper.get("paper_id", ""),
                paper.get("title", ""),
                json.dumps(paper.get("authors", [])),
                paper.get("year"),
                paper.get("arxiv_id"),
                paper.get("doi"),
                paper.get("abstract", ""),
                paper.get("source", "local"),
                paper.get("citation_count", 0) or 0,
                paper.get("influential_citation_count", 0) or 0,
                paper_type,
            ),
        )

    # Load quality scores if available
    quality_dir = cfg.get("paths", {}).get("quality_dir", "corpus/nuggets_quality") if cfg else None
    quality_scores = {}
    if quality_dir and os.path.isdir(quality_dir):
        for fname in os.listdir(quality_dir):
            if not fname.endswith(".json"):
                continue
            qdata = load_json(os.path.join(quality_dir, fname))
            for rated in qdata.get("scores", qdata.get("rated_nuggets", [])):
                nid = rated.get("nugget_id", "")
                if nid:
                    quality_scores[nid] = {
                        "overall_score": rated.get("overall_score"),
                        "flagged": 1 if (rated.get("overall_score") or 5) <= (
                            cfg.get("nuggets", {}).get("quality", {}).get("flag_threshold", 2)
                            if cfg else 2
                        ) else 0,
                    }
        if quality_scores:
            print(f"  Loaded quality scores for {len(quality_scores)} nuggets")

    # Insert nuggets
    flag_threshold = (cfg.get("nuggets", {}).get("quality", {}).get("flag_threshold", 2)
                      if cfg else 2)
    for n in nuggets:
        nid = n.get("nugget_id", "")
        # Prefer inlined quality (from unified pipeline)
        if "quality" in n and isinstance(n["quality"], dict):
            overall_score = n["quality"].get("overall")
            flagged = 1 if (overall_score if overall_score is not None else 5) <= flag_threshold else 0
        else:
            qs = quality_scores.get(nid, {})
            overall_score = qs.get("overall_score")
            flagged = qs.get("flagged", 0)
        c.execute(
            "INSERT OR IGNORE INTO nuggets VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                nid,
                n.get("paper_id", ""),
                n.get("question", ""),
                n.get("answer", ""),
                n.get("type", ""),
                n.get("confidence", ""),
                n.get("section", ""),
                n.get("source_chunk"),
                n.get("thesis_relevance", 0),
                overall_score,
                flagged,
                n.get("source_file"),
            ),
        )

    # Populate FTS5 index
    c.execute("""
        INSERT INTO nuggets_fts(nugget_id, question, answer)
        SELECT nugget_id, question, answer FROM nuggets
    """)

    # Populate code_links from JSON manifest if available
    code_links_path = cfg.get("paths", {}).get("code_links", "code_links.json") if cfg else None
    if code_links_path:
        # Resolve relative to project root (parent of kb_dir)
        if not os.path.isabs(code_links_path):
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(kb_dir)))
            code_links_path = os.path.join(project_root, code_links_path)
        if os.path.exists(code_links_path):
            links = load_json(code_links_path)
            for link in links:
                c.execute(
                    "INSERT INTO code_links (paper_id, source_file, link_type, description) VALUES (?,?,?,?)",
                    (link["paper_id"], link["source_file"], link["link_type"], link.get("description", "")),
                )
            print(f"  Loaded {len(links)} code-paper cross-references")

    conn.commit()
    papers_count = c.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    nuggets_count = c.execute("SELECT COUNT(*) FROM nuggets").fetchone()[0]
    fts_count = c.execute("SELECT COUNT(*) FROM nuggets_fts").fetchone()[0]
    conn.close()
    print(f"  SQLite {db_path}: {papers_count} papers, {nuggets_count} nuggets, {fts_count} FTS entries")


def _get_existing_paper_nugget_counts(db_path):
    """Get {paper_id: nugget_count} and {paper_id: content_hash} from existing SQLite DB."""
    import hashlib
    from collections import defaultdict

    if not os.path.exists(db_path):
        return {}, {}
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT paper_id, COUNT(*) as cnt FROM nuggets GROUP BY paper_id"
        ).fetchall()
        counts = {r["paper_id"]: r["cnt"] for r in rows}

        # Build per-paper content fingerprints
        fingerprints = defaultdict(list)
        for row in conn.execute("SELECT paper_id, nugget_id, question, answer FROM nuggets"):
            fingerprints[row["paper_id"]].append(
                f"{row['nugget_id']}:{row['question']}:{row['answer']}"
            )
        hashes = {}
        for pid, parts in fingerprints.items():
            hashes[pid] = hashlib.md5("||".join(sorted(parts)).encode()).hexdigest()

        return counts, hashes
    except sqlite3.OperationalError:
        return {}, {}  # table doesn't exist
    finally:
        conn.close()


def _detect_changed_papers(new_nuggets, existing_counts, existing_fingerprints=None):
    """Determine which papers need updating.

    Returns (changed_paper_ids, unchanged_paper_ids).
    A paper is changed if it's new, its nugget count differs, or its content changed.
    """
    import hashlib
    from collections import Counter, defaultdict

    if existing_fingerprints is None:
        existing_fingerprints = {}

    new_counts = Counter(n.get("paper_id", "") for n in new_nuggets)
    all_papers = set(new_counts.keys()) | set(existing_counts.keys())

    # Build per-paper content fingerprint from new nuggets
    new_fingerprints = defaultdict(list)
    for n in new_nuggets:
        pid = n.get("paper_id", "")
        nid = n.get("nugget_id", "")
        q = n.get("question", "")
        a = n.get("answer", "")
        new_fingerprints[pid].append(f"{nid}:{q}:{a}")

    changed = set()
    unchanged = set()
    for pid in all_papers:
        if new_counts.get(pid, 0) != existing_counts.get(pid, 0):
            changed.add(pid)
        elif pid in existing_fingerprints and pid in new_fingerprints:
            # Same count — check content hash
            new_hash = hashlib.md5("||".join(sorted(new_fingerprints[pid])).encode()).hexdigest()
            if new_hash != existing_fingerprints.get(pid, ""):
                changed.add(pid)
            else:
                unchanged.add(pid)
        else:
            unchanged.add(pid)
    return changed, unchanged


def update_chromadb(nuggets, embeddings, kb_dir, collection_name, changed_papers, distance_fn="cosine"):
    """Incrementally update ChromaDB for changed papers only."""
    chroma_path = os.path.join(kb_dir, "chromadb")
    os.makedirs(chroma_path, exist_ok=True)
    client = chromadb.PersistentClient(path=chroma_path)

    try:
        collection = client.get_collection(name=collection_name)
    except (ValueError, Exception):
        # Collection doesn't exist — fall back to full build
        return build_chromadb(nuggets, embeddings, kb_dir, collection_name, distance_fn)

    # Delete old nuggets for changed papers
    for pid in changed_papers:
        try:
            collection.delete(where={"paper_id": pid})
        except Exception:
            pass  # paper may not exist yet

    # Insert new nuggets for changed papers only
    changed_indices = [
        i for i, n in enumerate(nuggets) if n.get("paper_id", "") in changed_papers
    ]
    if not changed_indices:
        print(f"  ChromaDB: no nuggets to update")
        return collection

    batch_size = 500
    for start in range(0, len(changed_indices), batch_size):
        end = min(start + batch_size, len(changed_indices))
        batch_idx = changed_indices[start:end]
        batch_nuggets = [nuggets[i] for i in batch_idx]
        batch_embs = embeddings[batch_idx].tolist()

        ids = [n["nugget_id"] for n in batch_nuggets]
        documents = [
            f"[{n.get('type', '')}] [{n.get('section', '')}] Q: {n['question']} A: {n['answer']}"
            for n in batch_nuggets
        ]
        metadatas = [
            {
                "paper_id": n.get("paper_id", ""),
                "type": n.get("type", ""),
                "confidence": n.get("confidence", ""),
                "section": n.get("section", ""),
                "thesis_relevance": n.get("thesis_relevance", 3),
                "source_file": n.get("source_file", ""),
            }
            for n in batch_nuggets
        ]

        collection.upsert(
            ids=ids,
            embeddings=batch_embs,
            documents=documents,
            metadatas=metadatas,
        )

    print(f"  ChromaDB: updated {len(changed_indices)} nuggets across {len(changed_papers)} papers "
          f"(collection total: {collection.count()})")
    return collection


def update_sqlite(nuggets, manifest, kb_dir, changed_papers, db_name="nuggets.db", cfg=None):
    """Incrementally update SQLite for changed papers only."""
    db_path = os.path.join(kb_dir, db_name)
    if not os.path.exists(db_path):
        # No existing DB — fall back to full build
        return build_sqlite(nuggets, manifest, kb_dir, db_name, cfg=cfg)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    flag_threshold = (cfg.get("nuggets", {}).get("quality", {}).get("flag_threshold", 2)
                      if cfg else 2)

    # Delete old nuggets for changed papers via explicit FTS delete commands
    # followed by row deletion from the main table
    for pid in changed_papers:
        # Get rowids for FTS deletion (content-sync table needs explicit delete commands)
        rows = c.execute(
            "SELECT rowid, nugget_id, question, answer FROM nuggets WHERE paper_id = ?", (pid,)
        ).fetchall()
        for row in rows:
            c.execute(
                "INSERT INTO nuggets_fts(nuggets_fts, rowid, nugget_id, question, answer) "
                "VALUES('delete', ?, ?, ?, ?)",
                (row["rowid"], row["nugget_id"], row["question"], row["answer"]),
            )
        c.execute("DELETE FROM nuggets WHERE paper_id = ?", (pid,))

    # Remove papers that have no nuggets left (deleted entirely)
    manifest_pids = {p.get("paper_id", "") for p in manifest}
    for pid in changed_papers:
        if pid not in manifest_pids:
            c.execute("DELETE FROM papers WHERE paper_id = ?", (pid,))

    # Upsert papers from manifest
    for paper in manifest:
        pid = paper.get("paper_id", "")
        if pid not in changed_papers:
            continue
        pub_types = paper.get("publication_types", [])
        paper_type = ",".join(pub_types) if isinstance(pub_types, list) else str(pub_types or "")
        c.execute(
            "INSERT OR REPLACE INTO papers VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                pid,
                paper.get("title", ""),
                json.dumps(paper.get("authors", [])),
                paper.get("year"),
                paper.get("arxiv_id"),
                paper.get("doi"),
                paper.get("abstract", ""),
                paper.get("source", "local"),
                paper.get("citation_count", 0) or 0,
                paper.get("influential_citation_count", 0) or 0,
                paper_type,
            ),
        )

    # Insert new nuggets for changed papers
    changed_nuggets = [n for n in nuggets if n.get("paper_id", "") in changed_papers]
    for n in changed_nuggets:
        nid = n.get("nugget_id", "")
        if "quality" in n and isinstance(n["quality"], dict):
            overall_score = n["quality"].get("overall")
            flagged = 1 if (overall_score if overall_score is not None else 5) <= flag_threshold else 0
        else:
            overall_score = None
            flagged = 0
        c.execute(
            "INSERT OR IGNORE INTO nuggets VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                nid,
                n.get("paper_id", ""),
                n.get("question", ""),
                n.get("answer", ""),
                n.get("type", ""),
                n.get("confidence", ""),
                n.get("section", ""),
                n.get("source_chunk"),
                n.get("thesis_relevance", 0),
                overall_score,
                flagged,
                n.get("source_file"),
            ),
        )

    # Insert FTS entries for new nuggets
    for n in changed_nuggets:
        row = c.execute(
            "SELECT rowid FROM nuggets WHERE nugget_id = ?", (n.get("nugget_id", ""),)
        ).fetchone()
        if row:
            c.execute(
                "INSERT INTO nuggets_fts(rowid, nugget_id, question, answer) VALUES(?, ?, ?, ?)",
                (row["rowid"], n.get("nugget_id", ""), n.get("question", ""), n.get("answer", "")),
            )

    conn.commit()
    total = c.execute("SELECT COUNT(*) FROM nuggets").fetchone()[0]
    conn.close()
    print(f"  SQLite: updated {len(changed_nuggets)} nuggets across {len(changed_papers)} papers "
          f"(total: {total})")


def run_build(config_path="config.yaml", incremental=False):
    """Build the complete knowledge base.

    Args:
        incremental: If True, only update papers with changed nuggets.
            Compares nugget counts per paper against existing SQLite DB.
    """
    cfg = load_config(config_path)
    kb_dir = cfg["paths"]["kb_dir"]
    corpus_dir = cfg["paths"]["corpus_dir"]
    store_cfg = cfg.get("store", {})
    chroma_cfg = store_cfg.get("chromadb", {})
    sqlite_cfg = store_cfg.get("sqlite", {})
    os.makedirs(kb_dir, exist_ok=True)

    # Load data
    print("[store] Loading data...")
    nug_path = os.path.join(kb_dir, "nuggets_with_embeddings.jsonl")
    emb_path = os.path.join(kb_dir, "embeddings.npy")
    manifest_path = os.path.join(corpus_dir, "manifest.json")

    if not os.path.exists(nug_path) or not os.path.exists(emb_path):
        print("Missing nuggets_with_embeddings.jsonl or embeddings.npy. Run embed first.")
        return

    nuggets = load_jsonl(nug_path)
    embeddings = np.load(emb_path)
    manifest = load_json(manifest_path) if os.path.exists(manifest_path) else []
    print(f"  {len(nuggets)} nuggets, {embeddings.shape} embeddings, {len(manifest)} papers")

    if len(nuggets) != embeddings.shape[0]:
        print(f"ERROR: nugget count ({len(nuggets)}) != embedding rows ({embeddings.shape[0]}). Re-run embed.")
        return

    collection_name = chroma_cfg.get("collection_name", "thesis_nuggets")
    distance_fn = chroma_cfg.get("distance_fn", "cosine")
    db_name = sqlite_cfg.get("db_name", "nuggets.db")

    if incremental:
        db_path = os.path.join(kb_dir, db_name)
        existing_counts, existing_fingerprints = _get_existing_paper_nugget_counts(db_path)
        if not existing_counts:
            print("[store] No existing DB found, falling back to full build")
            incremental = False
        else:
            changed, unchanged = _detect_changed_papers(nuggets, existing_counts, existing_fingerprints)
            print(f"[store] Incremental: {len(changed)} papers changed, {len(unchanged)} unchanged")
            if not changed:
                print("[store] Nothing to update.")
                return

    if incremental:
        print("[store] Updating ChromaDB...")
        update_chromadb(nuggets, embeddings, kb_dir, collection_name, changed, distance_fn)
        print("[store] Updating SQLite...")
        update_sqlite(nuggets, manifest, kb_dir, changed, db_name, cfg=cfg)
    else:
        print("[store] Building ChromaDB...")
        build_chromadb(nuggets, embeddings, kb_dir, collection_name, distance_fn)
        print("[store] Building SQLite...")
        build_sqlite(nuggets, manifest, kb_dir, db_name, cfg=cfg)

    print("\nKB build complete.")


def main():
    ap = argparse.ArgumentParser(description="Build knowledge base")
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("--incremental", action="store_true",
                    help="Only update papers with changed nuggets")
    args = ap.parse_args()
    run_build(args.config, incremental=args.incremental)


if __name__ == "__main__":
    main()
