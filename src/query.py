"""Query interface for the thesis knowledge base."""
import os, sys, argparse, sqlite3, json, re
from pathlib import Path
import chromadb
from src.utils import load_config
from src.embed.embedder import make_embed_client, format_nugget_text
from src.log import get_logger

log = get_logger("query", "query.log")


class ThesisKB:
    """Query interface over ChromaDB + SQLite."""

    def __init__(self, config_path="config.yaml"):
        cfg = load_config(config_path)
        self._config_path = config_path
        kb_dir = cfg["paths"]["kb_dir"]
        store_cfg = cfg.get("store", {})
        chroma_cfg = store_cfg.get("chromadb", {})
        sqlite_cfg = store_cfg.get("sqlite", {})
        emb_cfg = cfg.get("embed", {}).get("embedding", {})

        self.db = None  # ensure attribute exists for close()

        # Embedding client for query-time embedding
        self.embed_client, self.embed_model = make_embed_client(cfg)
        self.embed_instruction = emb_cfg.get("instruction", "")
        self.embed_dimensions = emb_cfg.get("dimensions", None)

        # ChromaDB (may fail — open before SQLite)
        chroma_path = os.path.join(kb_dir, "chromadb")
        self.chroma = chromadb.PersistentClient(path=chroma_path)
        collection_name = chroma_cfg.get("collection_name", "thesis_nuggets")
        self.collection = self.chroma.get_collection(collection_name)

        # SQLite (opened last so earlier failures don't leak the connection)
        db_path = os.path.join(kb_dir, sqlite_cfg.get("db_name", "nuggets.db"))
        self.db = sqlite3.connect(db_path)
        self.db.row_factory = sqlite3.Row

        # Preload paper nugget counts for depth-of-coverage scoring
        rows = self.db.execute(
            "SELECT paper_id, COUNT(*) FROM nuggets GROUP BY paper_id"
        ).fetchall()
        self._paper_nugget_counts = {r[0]: r[1] for r in rows}

        # Ensure FTS5 index and schema migrations exist (self-migrating)
        self._ensure_fts5()
        self._ensure_quality_columns()
        self._ensure_indexes()

    @classmethod
    def sqlite_only(cls, config_path="config.yaml"):
        """Open only the SQLite connection (no embedding server needed)."""
        cfg = load_config(config_path)
        kb_dir = cfg["paths"]["kb_dir"]
        sqlite_cfg = cfg.get("store", {}).get("sqlite", {})
        db_path = os.path.join(kb_dir, sqlite_cfg.get("db_name", "nuggets.db"))
        instance = object.__new__(cls)
        instance.db = sqlite3.connect(db_path)
        instance.db.row_factory = sqlite3.Row
        instance.embed_client = None
        instance.embed_model = None
        instance.embed_instruction = None
        instance.embed_dimensions = None
        instance.chroma = None
        instance.collection = None
        rows = instance.db.execute(
            "SELECT paper_id, COUNT(*) FROM nuggets GROUP BY paper_id"
        ).fetchall()
        instance._paper_nugget_counts = {r[0]: r[1] for r in rows}
        return instance

    def paper_nugget_count(self, paper_id):
        """Return total nugget count for a paper (preloaded)."""
        return self._paper_nugget_counts.get(paper_id, 0)

    def _ensure_fts5(self):
        """Create FTS5 index if it doesn't exist (self-migrating)."""
        if not self.db:
            return
        exists = self.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='nuggets_fts'"
        ).fetchone()
        if exists:
            return
        self.db.execute("""
            CREATE VIRTUAL TABLE nuggets_fts USING fts5(
                nugget_id UNINDEXED,
                question,
                answer,
                content='nuggets',
                content_rowid='rowid'
            )
        """)
        self.db.execute("""
            INSERT INTO nuggets_fts(nugget_id, question, answer)
            SELECT nugget_id, question, answer FROM nuggets
        """)
        self.db.commit()

    def _ensure_quality_columns(self):
        """Add quality score columns if missing (self-migrating)."""
        if not self.db:
            return
        cols = {row[1] for row in self.db.execute("PRAGMA table_info(nuggets)").fetchall()}
        if "overall_score" not in cols:
            self.db.execute("ALTER TABLE nuggets ADD COLUMN overall_score REAL DEFAULT NULL")
            log.info("Migrated: added overall_score column")
        if "flagged" not in cols:
            self.db.execute("ALTER TABLE nuggets ADD COLUMN flagged INTEGER DEFAULT 0")
            log.info("Migrated: added flagged column")
            self.db.commit()

    def _ensure_indexes(self):
        """Create missing indexes (self-migrating)."""
        if not self.db:
            return
        existing = {row[1] for row in self.db.execute(
            "SELECT * FROM sqlite_master WHERE type='index'"
        ).fetchall()}
        indexes = {
            "idx_nuggets_relevance": "CREATE INDEX idx_nuggets_relevance ON nuggets(thesis_relevance)",
            "idx_nuggets_section": "CREATE INDEX idx_nuggets_section ON nuggets(section)",
            "idx_nuggets_paper_type": "CREATE INDEX idx_nuggets_paper_type ON nuggets(paper_id, type)",
            "idx_nuggets_flagged": "CREATE INDEX idx_nuggets_flagged ON nuggets(flagged)",
        }
        for name, ddl in indexes.items():
            if name not in existing:
                try:
                    self.db.execute(ddl)
                    log.info("Migrated: created index %s", name)
                except Exception as e:
                    log.warning("Failed to create index %s: %s", name, e)
        self.db.commit()

    @staticmethod
    def _sanitize_fts5_query(query: str) -> str:
        """Escape FTS5 special syntax so user input is treated as plain terms.

        Wraps each token in double-quotes to neutralise operators
        (AND, OR, NOT, NEAR, *, ^) and other FTS5 metacharacters.
        """
        # Strip characters that cannot appear even inside quoted strings
        query = re.sub(r'["\']', " ", query)
        tokens = query.split()
        if not tokens:
            return '""'
        return " ".join(f'"{t}"' for t in tokens)

    def bm25_search(self, query, n_results=20):
        """BM25 full-text search over nuggets via FTS5.

        Returns list of (nugget_id, bm25_score) tuples, best first.
        """
        if not self.db:
            return []
        sanitized = self._sanitize_fts5_query(query)
        try:
            rows = self.db.execute(
                """SELECT nugget_id, rank
                   FROM nuggets_fts
                   WHERE nuggets_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (sanitized, n_results),
            ).fetchall()
            # rank is negative (lower = better), convert to positive score
            return [(r[0], -r[1]) for r in rows]
        except Exception as e:
            log.warning("BM25 search failed for query %r: %s", query, e)
            return []

    def load_chunk(self, paper_id, chunk_id):
        """Load a specific chunk's text from disk."""
        cfg = load_config(self._config_path) if hasattr(self, '_config_path') else None
        chunk_dir = cfg["paths"]["chunk_dir"] if cfg else "corpus/chunks"
        chunk_path = Path(chunk_dir) / f"{paper_id}.json"
        if not chunk_path.exists():
            return None
        try:
            import json as _json
            data = _json.loads(chunk_path.read_text())
            for chunk in data.get("chunks", []):
                if chunk.get("chunk_id") == chunk_id:
                    return chunk.get("text", "")
        except Exception as e:
            log.warning("Failed to load chunk %s/%s: %s", paper_id, chunk_id, e)
        return None

    def _embed_query(self, text):
        """Embed a query string using the configured embedding backend."""
        formatted = format_nugget_text({"question": text, "answer": ""}, self.embed_instruction)
        kwargs = {"model": self.embed_model, "input": [formatted]}
        if self.embed_dimensions:
            kwargs["dimensions"] = self.embed_dimensions
        resp = self.embed_client.embeddings.create(**kwargs)
        return resp.data[0].embedding

    def query(self, text, n_results=10, type_filter=None, section_filter=None,
              types=None, year_min=None, year_max=None):
        """Query the KB with natural language text.

        Args:
            text: Query string.
            n_results: Max results to return.
            type_filter: Single type string (legacy).
            section_filter: Section string filter.
            types: List of types to include (e.g. ['method', 'result']).
            year_min: Minimum paper year (inclusive).
            year_max: Maximum paper year (inclusive).
        """
        # Build ChromaDB where clause
        where_clauses = []
        if types and len(types) == 1:
            where_clauses.append({"type": types[0]})
        elif types and len(types) > 1:
            where_clauses.append({"type": {"$in": types}})
        elif type_filter:
            where_clauses.append({"type": type_filter})
        if section_filter:
            where_clauses.append({"section": section_filter})

        where = None
        if len(where_clauses) == 1:
            where = where_clauses[0]
        elif len(where_clauses) > 1:
            where = {"$and": where_clauses}

        t0 = __import__("time").time()
        query_embedding = self._embed_query(text)
        log.debug("Embedded query in %.0fms", (__import__("time").time() - t0) * 1000)

        # Fetch extra results if we need to post-filter by year
        fetch_n = n_results * 3 if (year_min or year_max) else n_results

        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": fetch_n,
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        # Enrich with paper metadata and apply year filter
        enriched = []
        for i in range(len(results["ids"][0])):
            nugget_id = results["ids"][0][i]
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            doc = results["documents"][0][i]

            paper_id = meta.get("paper_id", "")
            paper = self._get_paper(paper_id)
            year = paper.get("year") if paper else None

            # Year filter
            if year_min and (not year or year < year_min):
                continue
            if year_max and (not year or year > year_max):
                continue

            enriched.append({
                "nugget_id": nugget_id,
                "distance": round(distance, 4),
                "type": meta.get("type", ""),
                "confidence": meta.get("confidence", ""),
                "section": meta.get("section", ""),
                "document": doc,
                "paper_id": paper_id,
                "paper_title": paper.get("title", "") if paper else "",
                "paper_year": year,
                "paper_authors": paper.get("authors", "") if paper else "",
                "arxiv_id": paper.get("arxiv_id", "") if paper else "",
            })
            if len(enriched) >= n_results:
                break

        return enriched

    def multi_query(self, queries, n_results=10, **kwargs):
        """Run multiple queries and return deduplicated results."""
        seen = set()
        all_results = []
        per_query = max(5, -(-n_results // len(queries)) + 3)  # ceil division + overlap buffer
        for q in queries:
            results = self.query(q, n_results=per_query, **kwargs)
            for r in results:
                if r["nugget_id"] not in seen:
                    seen.add(r["nugget_id"])
                    all_results.append(r)
        # Sort by distance (best first)
        all_results.sort(key=lambda x: x["distance"])
        return all_results[:n_results]

    def _get_paper(self, paper_id):
        """Look up paper metadata from SQLite."""
        row = self.db.execute(
            "SELECT * FROM papers WHERE paper_id = ?", (paper_id,)
        ).fetchone()
        if row:
            return dict(row)
        return None

    def find_papers(self, author=None, title=None, year=None, limit=10):
        """Search papers by author name, title substring, or year (SQLite only)."""
        conditions = []
        params = []
        if author:
            conditions.append("authors LIKE ?")
            params.append(f"%{author}%")
        if title:
            conditions.append("title LIKE ?")
            params.append(f"%{title}%")
        if year:
            conditions.append("year = ?")
            params.append(year)
        if not conditions:
            return []
        where = " AND ".join(conditions)
        rows = self.db.execute(
            f"SELECT * FROM papers WHERE {where} LIMIT ?",
            params + [limit]
        ).fetchall()
        return [dict(r) for r in rows]

    def get_paper_nuggets(self, paper_id, types=None):
        """Get all nuggets for a specific paper_id (SQLite only)."""
        query = "SELECT * FROM nuggets WHERE paper_id = ?"
        params = [paper_id]
        if types:
            placeholders = ",".join("?" * len(types))
            query += f" AND type IN ({placeholders})"
            params.extend(types)
        rows = self.db.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def stats(self):
        """Return KB statistics."""
        n_nuggets = self.collection.count()
        n_papers = self.db.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        types = self.db.execute(
            "SELECT type, COUNT(*) FROM nuggets GROUP BY type"
        ).fetchall()
        years = self.db.execute(
            "SELECT year, COUNT(*) FROM papers WHERE year IS NOT NULL GROUP BY year ORDER BY year"
        ).fetchall()
        sections = self.db.execute(
            "SELECT section, COUNT(*) FROM nuggets GROUP BY section ORDER BY COUNT(*) DESC"
        ).fetchall()
        return {
            "total_nuggets": n_nuggets,
            "total_papers": n_papers,
            "nuggets_by_type": {row[0]: row[1] for row in types},
            "papers_by_year": {row[0]: row[1] for row in years},
            "nuggets_by_section": {row[0]: row[1] for row in sections},
        }

    def close(self):
        if self.db is not None:
            self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def main():
    ap = argparse.ArgumentParser(description="Query thesis KB")
    ap.add_argument("query", nargs="?", help="Query text")
    ap.add_argument("-n", "--num", type=int, default=10, help="Number of results")
    ap.add_argument("-t", "--type", help="Filter by nugget type (single)")
    ap.add_argument("--types", help="Filter by nugget types (comma-separated, e.g. method,result)")
    ap.add_argument("-s", "--section", help="Filter by section")
    ap.add_argument("--year-min", type=int, help="Minimum paper year (inclusive)")
    ap.add_argument("--year-max", type=int, help="Maximum paper year (inclusive)")
    ap.add_argument("--queries", nargs="+", help="Multiple queries (deduplicated union)")
    ap.add_argument("--json", action="store_true", help="Output as JSON")
    ap.add_argument("--stats", action="store_true", help="Show KB stats")
    ap.add_argument("--find-author", help="Find papers by author name (SQLite, no embedding server)")
    ap.add_argument("--find-title", help="Find papers by title substring (SQLite, no embedding server)")
    ap.add_argument("--find-year", type=int, help="Find papers by year (use with --find-author/--find-title)")
    ap.add_argument("--paper-nuggets", help="Get all nuggets for a paper_id (SQLite, no embedding server)")
    ap.add_argument("--api", action="store_true",
                    help="Route queries through the API's full retrieval pipeline (requires running server)")
    ap.add_argument("--api-url", default="http://127.0.0.1:8001",
                    help="API server URL (default: http://127.0.0.1:8001)")
    ap.add_argument("--mode", default="survey",
                    help="Retrieval mode when using --api (background, draft, survey, etc.)")
    ap.add_argument("-c", "--config", default="config.yaml")
    args = ap.parse_args()

    # API-routed retrieval (full pipeline: expansion, RRF, reranking, diversity caps)
    if args.api:
        import urllib.request, urllib.error
        query_text = " ".join(args.queries) if args.queries else (args.query or "")
        if not query_text:
            print("Error: --api requires a query", file=sys.stderr)
            sys.exit(1)
        payload = json.dumps({
            "query": query_text,
            "mode": args.mode,
            "n_context": args.num,
        }).encode()
        req = urllib.request.Request(
            f"{args.api_url}/retrieve",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
        except urllib.error.URLError as e:
            print(f"Error: Cannot reach API at {args.api_url} — {e}", file=sys.stderr)
            print("Start the server first: bash scripts/start_local.sh", file=sys.stderr)
            sys.exit(1)

        results = data.get("results", [])
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            for i, r in enumerate(results):
                arxiv = f" arXiv:{r['arxiv_id']}" if r.get("arxiv_id") else ""
                print(f"\n--- [{i+1}] score={r.get('rrf_score', 0):.4f} type={r['type']} [{r.get('section', '')}] ---")
                print(f"Paper: {r.get('paper_title', '')} ({r.get('paper_year', '')}){arxiv}")
                print(f"Bibtex: {r.get('bibtex_key', '')}")
                print(f"{r['document']}")
        return

    # Direct SQLite lookups (no embedding server needed)
    if args.find_author or args.find_title or args.paper_nuggets:
        kb = ThesisKB.sqlite_only(args.config)
        if args.paper_nuggets:
            types_list = args.types.split(",") if args.types else None
            nuggets = kb.get_paper_nuggets(args.paper_nuggets, types=types_list)
            paper = kb._get_paper(args.paper_nuggets)
            result = {"paper": paper, "nuggets": nuggets, "nugget_count": len(nuggets)}
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                if paper:
                    print(f"Paper: {paper.get('title')} ({paper.get('year')})")
                    print(f"Authors: {paper.get('authors')}")
                    print(f"paper_id: {args.paper_nuggets}")
                print(f"\n{len(nuggets)} nuggets:")
                for n in nuggets:
                    print(f"  [{n['type']}] {n['question']}")
        else:
            papers = kb.find_papers(
                author=args.find_author,
                title=args.find_title,
                year=args.find_year,
                limit=args.num,
            )
            if args.json:
                print(json.dumps(papers, indent=2))
            else:
                if not papers:
                    print("No papers found.")
                for p in papers:
                    arxiv = f" arXiv:{p['arxiv_id']}" if p.get("arxiv_id") else ""
                    print(f"\n{p['paper_id']}")
                    print(f"  {p['title']} ({p['year']}){arxiv}")
                    print(f"  Authors: {p['authors']}")
        kb.close()
        return

    kb = ThesisKB(args.config)

    if args.stats:
        s = kb.stats()
        if args.json:
            print(json.dumps(s, indent=2))
        else:
            print(f"Papers: {s['total_papers']}")
            print(f"Nuggets: {s['total_nuggets']}")
            print("By type:")
            for t, c in sorted(s["nuggets_by_type"].items()):
                print(f"  {t:15s} {c}")
            print("By year:")
            for y, c in sorted(s["papers_by_year"].items()):
                print(f"  {y}  {c}")
        kb.close()
        return

    types_list = args.types.split(",") if args.types else None
    query_kwargs = dict(
        n_results=args.num,
        type_filter=args.type,
        section_filter=args.section,
        types=types_list,
        year_min=args.year_min,
        year_max=args.year_max,
    )

    if args.queries:
        results = kb.multi_query(args.queries, **query_kwargs)
    elif args.query:
        results = kb.query(args.query, **query_kwargs)
    else:
        print("Usage: python -m src.query \"your question\" [-n 10] [-t method] [--json]")
        kb.close()
        return

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for i, r in enumerate(results):
            arxiv = f" arXiv:{r['arxiv_id']}" if r.get("arxiv_id") else ""
            print(f"\n--- [{i+1}] dist={r['distance']} type={r['type']} [{r['section']}] ---")
            print(f"Paper: {r['paper_title']} ({r['paper_year']}){arxiv}")
            print(f"{r['document']}")

    kb.close()


if __name__ == "__main__":
    main()
