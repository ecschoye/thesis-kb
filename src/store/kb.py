"""Build ChromaDB + SQLite knowledge base from embedded nuggets."""
import os, json, sqlite3, argparse
import numpy as np
import chromadb
from src.utils import load_config, load_json


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
        documents = [f"Q: {n['question']} A: {n['answer']}" for n in batch_nuggets]
        metadatas = [
            {
                "paper_id": n.get("paper_id", ""),
                "type": n.get("type", ""),
                "confidence": n.get("confidence", ""),
                "section": n.get("section", ""),
                "thesis_relevance": n.get("thesis_relevance", 3),
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
            FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
        )
    """)

    c.execute("CREATE INDEX idx_nuggets_paper ON nuggets(paper_id)")
    c.execute("CREATE INDEX idx_nuggets_type ON nuggets(type)")
    c.execute("CREATE INDEX idx_nuggets_relevance ON nuggets(thesis_relevance)")
    c.execute("CREATE INDEX idx_nuggets_section ON nuggets(section)")
    c.execute("CREATE INDEX idx_nuggets_paper_type ON nuggets(paper_id, type)")
    c.execute("CREATE INDEX idx_nuggets_flagged ON nuggets(flagged)")

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
            for rated in qdata.get("rated_nuggets", []):
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
    for n in nuggets:
        nid = n.get("nugget_id", "")
        qs = quality_scores.get(nid, {})
        c.execute(
            "INSERT OR IGNORE INTO nuggets VALUES (?,?,?,?,?,?,?,?,?,?,?)",
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
                qs.get("overall_score"),
                qs.get("flagged", 0),
            ),
        )

    # Populate FTS5 index
    c.execute("""
        INSERT INTO nuggets_fts(nugget_id, question, answer)
        SELECT nugget_id, question, answer FROM nuggets
    """)

    conn.commit()
    papers_count = c.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    nuggets_count = c.execute("SELECT COUNT(*) FROM nuggets").fetchone()[0]
    fts_count = c.execute("SELECT COUNT(*) FROM nuggets_fts").fetchone()[0]
    conn.close()
    print(f"  SQLite {db_path}: {papers_count} papers, {nuggets_count} nuggets, {fts_count} FTS entries")


def run_build(config_path="config.yaml"):
    """Build the complete knowledge base."""
    cfg = load_config(config_path)
    kb_dir = cfg["paths"]["kb_dir"]
    corpus_dir = cfg["paths"]["corpus_dir"]
    store_cfg = cfg.get("store", {})
    chroma_cfg = store_cfg.get("chromadb", {})
    sqlite_cfg = store_cfg.get("sqlite", {})
    os.makedirs(kb_dir, exist_ok=True)

    # Load data
    print("[store] Loading data...")
    nug_path = os.path.join(kb_dir, "nuggets_with_embeddings.json")
    emb_path = os.path.join(kb_dir, "embeddings.npy")
    manifest_path = os.path.join(corpus_dir, "manifest.json")

    if not os.path.exists(nug_path) or not os.path.exists(emb_path):
        print("Missing nuggets_with_embeddings.json or embeddings.npy. Run embed first.")
        return

    nuggets = load_json(nug_path)
    embeddings = np.load(emb_path)
    manifest = load_json(manifest_path) if os.path.exists(manifest_path) else []
    print(f"  {len(nuggets)} nuggets, {embeddings.shape} embeddings, {len(manifest)} papers")

    if len(nuggets) != embeddings.shape[0]:
        print(f"ERROR: nugget count ({len(nuggets)}) != embedding rows ({embeddings.shape[0]}). Re-run embed.")
        return

    # Build ChromaDB
    print("[store] Building ChromaDB...")
    collection_name = chroma_cfg.get("collection_name", "thesis_nuggets")
    distance_fn = chroma_cfg.get("distance_fn", "cosine")
    build_chromadb(nuggets, embeddings, kb_dir, collection_name, distance_fn)

    # Build SQLite
    print("[store] Building SQLite...")
    db_name = sqlite_cfg.get("db_name", "nuggets.db")
    build_sqlite(nuggets, manifest, kb_dir, db_name, cfg=cfg)

    print("\nKB build complete.")


def main():
    ap = argparse.ArgumentParser(description="Build knowledge base")
    ap.add_argument("-c", "--config", default="config.yaml")
    args = ap.parse_args()
    run_build(args.config)


if __name__ == "__main__":
    main()
