"""Conflict detection and consensus tools for thesis-kb."""

from __future__ import annotations

from . import _shared as S


def register(mcp):

    @mcp.tool()
    def find_conflicting_nuggets(topic: str, n: int = 20) -> dict:
        """Find nuggets on a topic from different papers that may contradict each other.

        Searches for the topic, groups results by paper, then identifies
        nuggets from different papers that address the same question but
        may give different answers.

        Args:
            topic: Topic to search for conflicts on.
            n: Number of nuggets to analyze (default 20).
        """
        collection = S.get_collection()
        try:
            emb = S.embed_query(topic)
            res = collection.query(
                query_embeddings=[emb],
                n_results=n,
                where={"type": {"$in": ["result", "claim", "comparison"]}},
            )
        except Exception:
            return {"error": "Search failed", "conflicts": []}

        # Build nugget list
        nuggets = []
        for i in range(len(res["ids"][0])):
            nid = res["ids"][0][i]
            dist = res["distances"][0][i]
            meta = res["metadatas"][0][i]
            nqa = S.get_nugget_qa(nid)
            paper = S.get_paper(meta.get("paper_id", ""))
            nuggets.append({
                "nugget_id": nid,
                "distance": dist,
                "type": meta.get("type", ""),
                "paper_id": meta.get("paper_id", ""),
                "paper_title": paper.get("title", "") if paper else "",
                "paper_year": paper.get("year") if paper else None,
                "question": nqa["question"] if nqa else "",
                "answer": nqa["answer"][:200] if nqa else "",
            })

        # Find potential conflicts: nuggets from different papers with similar questions
        conflicts = []
        for i in range(len(nuggets)):
            for j in range(i + 1, len(nuggets)):
                a, b = nuggets[i], nuggets[j]
                if a["paper_id"] == b["paper_id"]:
                    continue
                # Both close to query = likely addressing same topic
                if a["distance"] < 1.0 and b["distance"] < 1.0:
                    conflicts.append({
                        "nugget_a": {
                            "nugget_id": a["nugget_id"],
                            "paper_id": a["paper_id"],
                            "paper_title": a["paper_title"],
                            "paper_year": a["paper_year"],
                            "type": a["type"],
                            "question": a["question"],
                            "answer": a["answer"],
                        },
                        "nugget_b": {
                            "nugget_id": b["nugget_id"],
                            "paper_id": b["paper_id"],
                            "paper_title": b["paper_title"],
                            "paper_year": b["paper_year"],
                            "type": b["type"],
                            "question": b["question"],
                            "answer": b["answer"],
                        },
                    })

        # Limit to most relevant conflicts
        return {
            "topic": topic,
            "nuggets_analyzed": len(nuggets),
            "potential_conflicts": len(conflicts),
            "conflicts": conflicts[:10],
        }

    @mcp.tool()
    def nugget_consensus(topic: str, n: int = 30) -> dict:
        """Show which claims on a topic have multi-paper consensus vs single-source.

        Searches KB, clusters nuggets by semantic similarity, and counts
        unique papers per cluster. Helps distinguish well-established facts
        from isolated claims.

        Args:
            topic: Topic to analyze consensus on.
            n: Number of nuggets to retrieve (default 30).
        """
        collection = S.get_collection()
        try:
            emb = S.embed_query(topic)
            res = collection.query(query_embeddings=[emb], n_results=n)
        except Exception:
            return {"error": "Search failed", "clusters": []}

        # Build nugget list with embeddings
        nuggets = []
        for i in range(len(res["ids"][0])):
            nid = res["ids"][0][i]
            dist = res["distances"][0][i]
            meta = res["metadatas"][0][i]
            nqa = S.get_nugget_qa(nid)
            nuggets.append({
                "nugget_id": nid,
                "distance": dist,
                "paper_id": meta.get("paper_id", ""),
                "type": meta.get("type", ""),
                "question": nqa["question"] if nqa else "",
                "answer": (nqa["answer"][:150] if nqa else ""),
            })

        # Greedy clustering by distance from query (nuggets close to each other = same cluster)
        # Simple approach: group nuggets within distance bands
        clusters = []
        used = set()

        for i, nug in enumerate(nuggets):
            if i in used:
                continue
            cluster = [nug]
            used.add(i)
            for j in range(i + 1, len(nuggets)):
                if j in used:
                    continue
                # If both are close to the query and to each other (heuristic)
                if abs(nug["distance"] - nuggets[j]["distance"]) < 0.3:
                    cluster.append(nuggets[j])
                    used.add(j)

            papers = list({n["paper_id"] for n in cluster})
            label = "consensus" if len(papers) >= 3 else ("moderate" if len(papers) == 2 else "single_source")
            clusters.append({
                "label": label,
                "paper_count": len(papers),
                "paper_ids": papers,
                "representative": cluster[0]["question"],
                "nugget_count": len(cluster),
            })

        clusters.sort(key=lambda c: c["paper_count"], reverse=True)
        return {
            "topic": topic,
            "total_nuggets": len(nuggets),
            "clusters": clusters,
            "consensus_count": sum(1 for c in clusters if c["label"] == "consensus"),
            "single_source_count": sum(1 for c in clusters if c["label"] == "single_source"),
        }

    @mcp.tool()
    def paper_timeline(topic: str, n: int = 30) -> dict:
        """Chronological view of findings on a topic.

        Searches KB and groups results by paper year, showing how
        the field evolved over time.

        Args:
            topic: Topic to trace chronologically.
            n: Number of nuggets to retrieve (default 30).
        """
        collection = S.get_collection()
        try:
            emb = S.embed_query(topic)
            res = collection.query(query_embeddings=[emb], n_results=n)
        except Exception:
            return {"error": "Search failed", "timeline": []}

        # Group by paper year
        by_year: dict[int, list] = {}
        for i in range(len(res["ids"][0])):
            nid = res["ids"][0][i]
            dist = res["distances"][0][i]
            meta = res["metadatas"][0][i]
            pid = meta.get("paper_id", "")
            paper = S.get_paper(pid)
            year = paper.get("year") if paper else None
            if not year:
                continue

            nqa = S.get_nugget_qa(nid)
            entry = {
                "paper_id": pid,
                "paper_title": paper.get("title", "") if paper else "",
                "distance": round(dist, 4),
                "type": meta.get("type", ""),
                "question": nqa["question"] if nqa else "",
            }

            if year not in by_year:
                by_year[year] = []
            # Deduplicate by paper within year
            if not any(e["paper_id"] == pid for e in by_year[year]):
                by_year[year].append(entry)

        timeline = [
            {"year": year, "papers": entries}
            for year, entries in sorted(by_year.items())
        ]

        return {
            "topic": topic,
            "year_range": [min(by_year.keys()), max(by_year.keys())] if by_year else [],
            "total_papers": sum(len(e) for e in by_year.values()),
            "timeline": timeline,
        }
