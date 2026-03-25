"""Code-aware tools for thesis-kb MCP server.

Provides tools for searching code nuggets, browsing codebase structure,
cross-referencing code files with academic papers, and querying
training configurations.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from . import _shared as S

CODE_PAPER_ID = "__smcm_mcfnet__"


def register(mcp):

    @mcp.tool()
    def code_search(
        query: str,
        n: int = 15,
        types: list[str] | None = None,
        file_filter: str | None = None,
    ) -> list[dict]:
        """Search code-derived nuggets from the SMCM-MCFNet implementation.

        Searches only nuggets extracted from the codebase (not academic papers).
        Use this to find implementation details, architecture decisions,
        training configs, and component relationships.

        Args:
            query: Natural language search query about the implementation.
            n: Max results (default 15).
            types: Filter by nugget type: implementation, config, experiment.
            file_filter: Filter to nuggets from a specific source file (substring match).
        """
        collection = S.get_collection()

        where_clauses = [{"paper_id": CODE_PAPER_ID}]
        if types and len(types) == 1:
            where_clauses.append({"type": types[0]})
        elif types and len(types) > 1:
            where_clauses.append({"type": {"$in": types}})

        where = where_clauses[0] if len(where_clauses) == 1 else {"$and": where_clauses}

        try:
            emb = S.embed_query(query)
            res = collection.query(
                query_embeddings=[emb],
                n_results=n * 2,
                where=where,
            )
        except Exception:
            # Fallback to BM25
            db = S.get_db()
            fts_q = S.sanitize_fts5_query(query)
            rows = db.execute(
                """SELECT n.nugget_id, n.question, n.answer, n.type, n.section,
                          n.source_file, n.paper_id
                   FROM nuggets_fts f
                   JOIN nuggets n ON f.nugget_id = n.nugget_id
                   WHERE nuggets_fts MATCH ? AND n.paper_id = ?
                   ORDER BY rank LIMIT ?""",
                (fts_q, CODE_PAPER_ID, n),
            ).fetchall()
            return [
                {
                    "nugget_id": r[0],
                    "question": r[1],
                    "answer": r[2],
                    "type": r[3],
                    "section": r[4],
                    "source_file": r[5],
                    "source": "bm25_fallback",
                }
                for r in rows
            ]

        results = []
        for i in range(len(res["ids"][0])):
            nid = res["ids"][0][i]
            meta = res["metadatas"][0][i]
            doc = res["documents"][0][i]
            dist = res["distances"][0][i]

            source_file = meta.get("source_file", "")
            if file_filter and file_filter not in source_file:
                continue

            qa = S.get_nugget_qa(nid)
            results.append({
                "nugget_id": nid,
                "distance": round(dist, 4),
                "type": meta.get("type", ""),
                "source_file": source_file,
                "question": qa["question"] if qa else doc.split(" A: ")[0].split("Q: ")[-1],
                "answer": qa["answer"] if qa else doc.split(" A: ")[-1],
            })
            if len(results) >= n:
                break

        return results

    @mcp.tool()
    def code_structure(module_path: str | None = None) -> dict:
        """Get the module hierarchy of the SMCM-MCFNet codebase.

        Returns classes, methods, and inheritance for all or a specific module.
        Uses cached AST extraction data.

        Args:
            module_path: Optional filter — substring match on module path
                         (e.g. "model/mcm", "backbone", "ofe").
        """
        cfg = S._cfg
        chunk_dir = cfg.get("paths", {}).get("code_chunk_dir", "corpus/code_chunks")
        chunk_path = Path(S._project_root) / chunk_dir / "code_chunks.json"

        if not chunk_path.exists():
            return {"error": "Code chunks not found. Run: python -m src.code_extract -c config.yaml"}

        with open(chunk_path) as f:
            chunks = json.load(f)

        if module_path:
            chunks = [c for c in chunks if module_path in c.get("file", "")]

        modules = []
        for chunk in chunks:
            mod = {
                "file": chunk["file"],
                "module": chunk["module"],
            }
            if chunk.get("module_doc"):
                mod["doc"] = chunk["module_doc"].split("\n")[0]

            classes = []
            for cls in chunk.get("classes", []):
                cls_info = {
                    "name": cls["name"],
                    "bases": cls["bases"],
                }
                if cls.get("docstring"):
                    cls_info["purpose"] = cls["docstring"].split("\n")[0]
                if cls.get("is_dataclass"):
                    cls_info["is_dataclass"] = True
                    cls_info["fields"] = [f["name"] for f in cls.get("dataclass_fields", [])]
                methods = [m["name"] for m in cls.get("methods", [])]
                if methods:
                    cls_info["methods"] = methods
                classes.append(cls_info)

            if classes:
                mod["classes"] = classes

            functions = [f["name"] for f in chunk.get("functions", [])]
            if functions:
                mod["functions"] = functions

            modules.append(mod)

        return {
            "total_modules": len(modules),
            "modules": modules,
        }

    @mcp.tool()
    def find_implementations(paper_id: str) -> list[dict]:
        """Find code files that implement methods from a given paper.

        Uses the code_links cross-reference table to map papers to
        implementation files in the SMCM-MCFNet codebase.

        Args:
            paper_id: Paper ID (e.g. "2103_06011" or full ID from KB).
        """
        db = S.get_db()
        # Check if code_links table exists
        tables = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='code_links'"
        ).fetchone()
        if not tables:
            return [{"error": "code_links table not found. Rebuild KB with latest schema."}]

        rows = db.execute(
            "SELECT source_file, link_type, description FROM code_links WHERE paper_id = ?",
            (paper_id,),
        ).fetchall()

        if not rows:
            # Try partial match
            rows = db.execute(
                "SELECT source_file, link_type, description, paper_id FROM code_links WHERE paper_id LIKE ?",
                (f"%{paper_id}%",),
            ).fetchall()
            if rows:
                return [
                    {
                        "source_file": r[0],
                        "link_type": r[1],
                        "description": r[2],
                        "paper_id": r[3],
                    }
                    for r in rows
                ]
            return []

        paper = S.get_paper(paper_id)
        return [
            {
                "source_file": r[0],
                "link_type": r[1],
                "description": r[2],
                "paper_id": paper_id,
                "paper_title": paper.get("title", "") if paper else "",
            }
            for r in rows
        ]

    @mcp.tool()
    def find_papers_for_code(source_file: str) -> list[dict]:
        """Find papers that inspired or are implemented by a given code file.

        Reverse lookup in the code_links cross-reference table.

        Args:
            source_file: Source file path or substring (e.g. "spiking_blocks.py", "mcm/mcm.py").
        """
        db = S.get_db()
        tables = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='code_links'"
        ).fetchone()
        if not tables:
            return [{"error": "code_links table not found. Rebuild KB with latest schema."}]

        rows = db.execute(
            "SELECT paper_id, link_type, description FROM code_links WHERE source_file LIKE ?",
            (f"%{source_file}%",),
        ).fetchall()

        results = []
        for r in rows:
            pid = r[0]
            paper = S.get_paper(pid)
            results.append({
                "paper_id": pid,
                "link_type": r[1],
                "description": r[2],
                "paper_title": paper.get("title", "") if paper else "",
                "paper_year": paper.get("year") if paper else None,
            })
        return results

    @mcp.tool()
    def implementation_detail(component: str, n: int = 10) -> dict:
        """Get both code implementation details and paper descriptions for a component.

        Searches for the component in both code nuggets and paper nuggets,
        enabling side-by-side comparison of "what the paper says" vs
        "how we implemented it."

        Args:
            component: Component name (e.g. "MCM", "EDUM", "ST-FlowNet",
                       "backbone", "spiking blocks", "event warping").
            n: Max results per source (default 10).
        """
        collection = S.get_collection()
        db = S.get_db()

        try:
            emb = S.embed_query(component)
        except Exception:
            # Fallback to BM25 only
            fts_q = S.sanitize_fts5_query(component)
            code_rows = db.execute(
                """SELECT nugget_id, question, answer, type, source_file
                   FROM nuggets WHERE paper_id = ? AND nugget_id IN
                   (SELECT nugget_id FROM nuggets_fts WHERE nuggets_fts MATCH ?)
                   LIMIT ?""",
                (CODE_PAPER_ID, fts_q, n),
            ).fetchall()
            paper_rows = db.execute(
                """SELECT n.nugget_id, n.question, n.answer, n.type, p.title, p.paper_id
                   FROM nuggets n JOIN papers p ON n.paper_id = p.paper_id
                   WHERE n.paper_id != ? AND n.nugget_id IN
                   (SELECT nugget_id FROM nuggets_fts WHERE nuggets_fts MATCH ?)
                   LIMIT ?""",
                (CODE_PAPER_ID, fts_q, n),
            ).fetchall()
            return {
                "component": component,
                "code_nuggets": [{"question": r[1], "answer": r[2], "type": r[3], "source_file": r[4]} for r in code_rows],
                "paper_nuggets": [{"question": r[1], "answer": r[2], "type": r[3], "paper_title": r[4], "paper_id": r[5]} for r in paper_rows],
                "source": "bm25_fallback",
            }

        # Code nuggets
        try:
            code_res = collection.query(
                query_embeddings=[emb],
                n_results=n,
                where={"paper_id": CODE_PAPER_ID},
            )
        except Exception:
            code_res = {"ids": [[]], "metadatas": [[]], "documents": [[]], "distances": [[]]}

        code_nuggets = []
        for i in range(len(code_res["ids"][0])):
            nid = code_res["ids"][0][i]
            meta = code_res["metadatas"][0][i]
            qa = S.get_nugget_qa(nid)
            code_nuggets.append({
                "question": qa["question"] if qa else "",
                "answer": qa["answer"] if qa else "",
                "type": meta.get("type", ""),
                "source_file": meta.get("source_file", ""),
                "distance": round(code_res["distances"][0][i], 4),
            })

        # Paper nuggets
        try:
            paper_res = collection.query(
                query_embeddings=[emb],
                n_results=n,
                where={"paper_id": {"$ne": CODE_PAPER_ID}},
            )
        except Exception:
            paper_res = {"ids": [[]], "metadatas": [[]], "documents": [[]], "distances": [[]]}

        paper_nuggets = []
        for i in range(len(paper_res["ids"][0])):
            nid = paper_res["ids"][0][i]
            meta = paper_res["metadatas"][0][i]
            qa = S.get_nugget_qa(nid)
            paper = S.get_paper(meta.get("paper_id", ""))
            paper_nuggets.append({
                "question": qa["question"] if qa else "",
                "answer": qa["answer"] if qa else "",
                "type": meta.get("type", ""),
                "paper_id": meta.get("paper_id", ""),
                "paper_title": paper.get("title", "") if paper else "",
                "distance": round(paper_res["distances"][0][i], 4),
            })

        # Cross-references
        code_links = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='code_links'"
        ).fetchone()
        links = []
        if code_links:
            link_rows = db.execute(
                """SELECT cl.paper_id, cl.source_file, cl.link_type, cl.description,
                          p.title as paper_title
                   FROM code_links cl
                   LEFT JOIN papers p ON cl.paper_id = p.paper_id
                   WHERE cl.source_file LIKE ? OR cl.description LIKE ?""",
                (f"%{component.lower()}%", f"%{component}%"),
            ).fetchall()
            links = [
                {
                    "paper_id": r[0],
                    "source_file": r[1],
                    "link_type": r[2],
                    "description": r[3],
                    "paper_title": r[4] or "",
                }
                for r in link_rows
            ]

        return {
            "component": component,
            "code_nuggets": code_nuggets,
            "paper_nuggets": paper_nuggets,
            "cross_references": links,
        }

    @mcp.tool()
    def training_config(run_name: str | None = None) -> dict:
        """Get training configuration for a specific run or list available runs.

        Parses YAML run configs from the SMCM-MCFNet config directory.

        Args:
            run_name: Name of the run config (e.g. "mcm_s_ann", "mcm_s_snn").
                      If None, returns list of available runs.
        """
        cfg = S._cfg
        chunk_dir = cfg.get("paths", {}).get("code_chunk_dir", "corpus/code_chunks")
        config_path = Path(S._project_root) / chunk_dir / "config_chunks.json"

        if not config_path.exists():
            return {"error": "Config chunks not found. Run: python -m src.code_extract -c config.yaml"}

        with open(config_path) as f:
            configs = json.load(f)

        if run_name is None:
            return {
                "available_runs": [
                    {
                        "name": c["run_name"],
                        "category": c.get("category", ""),
                        "file": c["file"],
                    }
                    for c in configs
                ],
            }

        # Find matching config
        matches = [c for c in configs if run_name.lower() in c["run_name"].lower()]
        if not matches:
            return {
                "error": f"No config found matching '{run_name}'",
                "available": [c["run_name"] for c in configs],
            }

        results = []
        for m in matches:
            results.append({
                "run_name": m["run_name"],
                "category": m.get("category", ""),
                "file": m["file"],
                "config": m.get("config", {}),
            })
        return {"configs": results}
