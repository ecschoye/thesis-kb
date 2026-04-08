"""Gap analysis and coverage tools for thesis-kb."""

from __future__ import annotations

from . import _shared as S
from . import _latex as L
from . import _text as T


def register(mcp):

    @mcp.tool()
    def find_citations_for_text(text: str, n: int = 5) -> list[dict]:
        """Decompose text into sentences, search KB for evidence per sentence.

        For each factual sentence, returns the best matching KB nuggets.
        Use to find citations for uncited claims in draft text.

        Args:
            text: Plain text or LaTeX to find citations for.
            n: Number of nuggets per sentence (default 5).
        """
        plain = L.strip_latex(text)
        sentences = T.split_sentences(plain)
        if not sentences:
            return []

        collection = S.get_collection()
        results = []

        for sent in sentences:
            # Skip very short or non-factual sentences
            words = sent.split()
            if len(words) < 5:
                continue

            try:
                emb = S.embed_query(sent)
                res = collection.query(query_embeddings=[emb], n_results=n)
            except Exception:
                continue

            matches = []
            for i in range(len(res["ids"][0])):
                nid = res["ids"][0][i]
                dist = res["distances"][0][i]
                nugget = S.get_nugget_qa(nid)
                paper = S.get_paper(nugget["paper_id"]) if nugget else None
                matches.append({
                    "nugget_id": nid,
                    "distance": round(dist, 4),
                    "type": nugget["type"] if nugget else "",
                    "question": nugget["question"] if nugget else "",
                    "answer": (nugget["answer"][:200] + "...") if nugget and len(nugget.get("answer", "")) > 200 else (nugget["answer"] if nugget else ""),
                    "paper_id": nugget["paper_id"] if nugget else "",
                    "paper_title": paper.get("title", "") if paper else "",
                    "paper_year": paper.get("year") if paper else None,
                })

            results.append({
                "sentence": sent,
                "matches": matches,
            })

        return results

    @mcp.tool()
    def section_gap_analysis(tex_content: str, n: int = 10) -> dict:
        """Find KB papers relevant to the section that are not currently cited.

        Extracts section topic from content, searches KB, then excludes
        papers already cited. Returns uncited papers ranked by relevance.

        Args:
            tex_content: LaTeX source text of the section.
            n: Number of uncited papers to return (default 10).
        """
        # Extract currently cited papers
        cite_keys = L.extract_all_cite_keys(tex_content)
        cited_paper_ids = set()
        for key in cite_keys:
            paper = S.resolve_cite_key(key)
            if paper:
                cited_paper_ids.add(paper["paper_id"])

        # Search KB with the section content
        plain = L.strip_latex(tex_content)
        # Use first 500 chars as topic query
        topic = plain[:500] if len(plain) > 500 else plain

        collection = S.get_collection()
        try:
            emb = S.embed_query(topic)
            res = collection.query(query_embeddings=[emb], n_results=n * 5)
        except Exception:
            return {"error": "Embedding failed", "uncited_papers": []}

        # Group by paper, exclude cited
        paper_scores: dict[str, dict] = {}
        for i in range(len(res["ids"][0])):
            nid = res["ids"][0][i]
            dist = res["distances"][0][i]
            meta = res["metadatas"][0][i]
            pid = meta.get("paper_id", "")

            if pid in cited_paper_ids:
                continue
            if pid not in paper_scores:
                paper = S.get_paper(pid)
                paper_scores[pid] = {
                    "paper_id": pid,
                    "title": paper.get("title", "") if paper else "",
                    "year": paper.get("year") if paper else None,
                    "authors": paper.get("authors", "") if paper else "",
                    "arxiv_id": paper.get("arxiv_id", "") if paper else "",
                    "best_distance": dist,
                    "nugget_hits": 0,
                    "best_nugget": None,
                }
            paper_scores[pid]["nugget_hits"] += 1
            if dist < paper_scores[pid]["best_distance"]:
                paper_scores[pid]["best_distance"] = dist
                nugget = S.get_nugget_qa(nid)
                paper_scores[pid]["best_nugget"] = {
                    "question": nugget["question"] if nugget else "",
                    "type": nugget["type"] if nugget else "",
                }

        # Sort by best distance
        uncited = sorted(paper_scores.values(), key=lambda x: x["best_distance"])[:n]
        for p in uncited:
            p["best_distance"] = round(p["best_distance"], 4)

        return {
            "cited_papers": len(cited_paper_ids),
            "uncited_relevant": len(uncited),
            "uncited_papers": uncited,
        }

    @mcp.tool()
    def suggest_coverage(
        topic: str,
        exclude_papers: list[str] | None = None,
        n: int = 10,
    ) -> list[dict]:
        """Suggest KB papers to strengthen coverage of a topic.

        Multi-signal ranking: number of relevant nuggets, best nugget distance,
        citation count, and recency. Excludes specified papers.

        Args:
            topic: Topic to find coverage for.
            exclude_papers: Paper IDs to exclude (already used).
            n: Number of suggestions (default 10).
        """
        exclude = set(exclude_papers or [])
        collection = S.get_collection()

        try:
            emb = S.embed_query(topic)
            res = collection.query(query_embeddings=[emb], n_results=n * 5)
        except Exception:
            return []

        paper_data: dict[str, dict] = {}
        for i in range(len(res["ids"][0])):
            dist = res["distances"][0][i]
            meta = res["metadatas"][0][i]
            pid = meta.get("paper_id", "")
            if pid in exclude:
                continue

            if pid not in paper_data:
                paper = S.get_paper(pid)
                if not paper:
                    continue
                paper_data[pid] = {
                    "paper_id": pid,
                    "title": paper.get("title", ""),
                    "year": paper.get("year"),
                    "authors": paper.get("authors", ""),
                    "arxiv_id": paper.get("arxiv_id", ""),
                    "citation_count": paper.get("citation_count", 0) or 0,
                    "best_distance": dist,
                    "nugget_hits": 0,
                }
            paper_data[pid]["nugget_hits"] += 1
            if dist < paper_data[pid]["best_distance"]:
                paper_data[pid]["best_distance"] = dist

        # Composite score: lower is better
        suggestions = list(paper_data.values())
        for s in suggestions:
            # Normalize: distance (0-2), recency bonus, citation bonus
            recency = max(0, (s["year"] or 2020) - 2015) / 10  # 0-1
            cite_norm = min(1, (s["citation_count"]) / 100)  # 0-1
            s["score"] = round(
                s["best_distance"] * 0.5
                - s["nugget_hits"] * 0.1
                - recency * 0.2
                - cite_norm * 0.2,
                4,
            )
            s["best_distance"] = round(s["best_distance"], 4)

        suggestions.sort(key=lambda x: x["score"])
        return suggestions[:n]

    @mcp.tool()
    def find_counter_evidence(claim: str, n: int = 10) -> list[dict]:
        """Find KB nuggets that contradict or qualify a given claim.

        Searches with the claim and negated variations, focusing on
        limitation and comparison nugget types.

        Args:
            claim: The claim to find counter-evidence for.
            n: Max results (default 10).
        """
        collection = S.get_collection()
        all_results = []
        seen = set()

        # Search strategies
        queries = [claim]
        # Add negated/qualifying variants
        queries.append(f"limitations of {claim}")
        queries.append(f"fails to {claim}")
        queries.append(f"unlike {claim}")

        for q in queries:
            try:
                emb = S.embed_query(q)
                res = collection.query(
                    query_embeddings=[emb],
                    n_results=n,
                    where={"type": {"$in": ["limitation", "comparison", "claim"]}},
                )
            except Exception:
                continue

            for i in range(len(res["ids"][0])):
                nid = res["ids"][0][i]
                if nid in seen:
                    continue
                seen.add(nid)
                dist = res["distances"][0][i]
                nugget = S.get_nugget_qa(nid)
                paper = S.get_paper(nugget["paper_id"]) if nugget else None
                all_results.append({
                    "nugget_id": nid,
                    "distance": round(dist, 4),
                    "type": nugget["type"] if nugget else "",
                    "question": nugget["question"] if nugget else "",
                    "answer": (nugget["answer"][:200] + "...") if nugget and len(nugget.get("answer", "")) > 200 else (nugget["answer"] if nugget else ""),
                    "paper_id": nugget["paper_id"] if nugget else "",
                    "paper_title": paper.get("title", "") if paper else "",
                    "paper_year": paper.get("year") if paper else None,
                })

        all_results.sort(key=lambda x: x["distance"])
        return all_results[:n]

    @mcp.tool()
    def compare_papers(paper_id_a: str, paper_id_b: str) -> dict:
        """Side-by-side comparison of two papers' nuggets grouped by type.

        For each nugget type, shows what each paper contributes.

        Args:
            paper_id_a: First paper ID.
            paper_id_b: Second paper ID.
        """
        db = S.get_db()
        paper_a = S.get_paper(paper_id_a)
        paper_b = S.get_paper(paper_id_b)

        if not paper_a or not paper_b:
            missing = []
            if not paper_a:
                missing.append(paper_id_a)
            if not paper_b:
                missing.append(paper_id_b)
            return {"error": f"Paper(s) not found: {', '.join(missing)}"}

        def get_nuggets(pid):
            rows = db.execute(
                "SELECT nugget_id, question, answer, type FROM nuggets WHERE paper_id = ?",
                (pid,),
            ).fetchall()
            return [dict(r) for r in rows]

        nuggets_a = get_nuggets(paper_id_a)
        nuggets_b = get_nuggets(paper_id_b)

        # Group by type
        types = {"method", "result", "claim", "limitation", "comparison", "background"}
        comparison = {}
        for ntype in types:
            a_typed = [n for n in nuggets_a if n["type"] == ntype][:5]
            b_typed = [n for n in nuggets_b if n["type"] == ntype][:5]
            if a_typed or b_typed:
                comparison[ntype] = {
                    "paper_a": [{"question": n["question"], "answer": n["answer"][:200]} for n in a_typed],
                    "paper_b": [{"question": n["question"], "answer": n["answer"][:200]} for n in b_typed],
                }

        return {
            "paper_a": {"paper_id": paper_id_a, "title": paper_a.get("title", ""), "year": paper_a.get("year")},
            "paper_b": {"paper_id": paper_id_b, "title": paper_b.get("title", ""), "year": paper_b.get("year")},
            "nugget_counts": {"paper_a": len(nuggets_a), "paper_b": len(nuggets_b)},
            "comparison": comparison,
        }
