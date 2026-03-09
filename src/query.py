"""Query interface for the thesis knowledge base."""
import os, argparse, sqlite3, json
import chromadb
from src.utils import load_config
from src.embed.embedder import make_embed_client, format_nugget_text


class ThesisKB:
    """Query interface over ChromaDB + SQLite."""

    def __init__(self, config_path="config.yaml"):
        cfg = load_config(config_path)
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

        query_embedding = self._embed_query(text)

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
        per_query = max(n_results, n_results // len(queries) + 5)
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
    ap.add_argument("-c", "--config", default="config.yaml")
    args = ap.parse_args()

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
