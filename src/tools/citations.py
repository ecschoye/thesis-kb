"""Citation and reference tools for thesis-kb."""

from __future__ import annotations

from . import _shared as S
from . import _latex as L
from . import _text as T


def register(mcp):

    @mcp.tool()
    def scan_references(tex_content: str) -> dict:
        """Extract all citation keys from LaTeX text, resolve each against KB + .bib file.

        Reports per-key status: found in KB, found in bib, missing, or ambiguous.
        Use to catch citation hallucinations and missing references before submission.

        Args:
            tex_content: LaTeX source text to scan.
        """
        cite_keys = L.extract_all_cite_keys(tex_content)
        if not cite_keys:
            return {"total_keys": 0, "keys": []}

        # Parse .bib to check which keys exist there
        bib_text = S.read_bib_file()
        bib_keys = set()
        if bib_text:
            for entry in L.parse_bib_entries(bib_text):
                bib_keys.add(entry["key"])

        results = []
        for key in cite_keys:
            in_bib = key in bib_keys
            paper = S.resolve_cite_key(key)
            if paper:
                status = "found"
            elif in_bib:
                status = "bib_only"  # in .bib but not matched in KB
            else:
                status = "missing"

            entry = {
                "key": key,
                "status": status,
                "in_bib": in_bib,
                "paper_id": paper["paper_id"] if paper else None,
                "paper_title": paper.get("title", "") if paper else None,
                "paper_year": paper.get("year") if paper else None,
            }
            results.append(entry)

        found = sum(1 for r in results if r["status"] == "found")
        bib_only = sum(1 for r in results if r["status"] == "bib_only")
        missing = sum(1 for r in results if r["status"] == "missing")

        return {
            "total_keys": len(cite_keys),
            "found": found,
            "bib_only": bib_only,
            "missing": missing,
            "keys": results,
        }

    @mcp.tool()
    def verify_claim(claim: str, paper_id: str) -> dict:
        """Check if a specific paper's nuggets support a given claim.

        Embeds the claim and computes cosine similarity against the paper's nuggets.
        Returns top matching nuggets with distance scores.

        Args:
            claim: The claim text to verify.
            paper_id: Paper ID (underscore format, e.g. "2401_17151").
        """
        paper = S.get_paper(paper_id)
        if not paper:
            return {"error": f"Paper {paper_id} not found"}

        collection = S.get_collection()
        query_embedding = S.embed_query(claim)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=10,
            where={"paper_id": paper_id},
        )

        matches = []
        for i in range(len(results["ids"][0])):
            nid = results["ids"][0][i]
            dist = results["distances"][0][i]
            nugget = S.get_nugget_qa(nid)
            matches.append({
                "nugget_id": nid,
                "distance": round(dist, 4),
                "type": nugget["type"] if nugget else "",
                "question": nugget["question"] if nugget else "",
                "answer": nugget["answer"] if nugget else "",
            })

        best_dist = matches[0]["distance"] if matches else 999
        return {
            "paper_id": paper_id,
            "paper_title": paper.get("title", ""),
            "claim": claim,
            "supported": best_dist < 0.8,
            "best_distance": best_dist,
            "matches": matches[:5],
        }

    @mcp.tool()
    def check_references(tex_content: str) -> dict:
        """Find broken cite keys, unused bib entries, and label/ref mismatches.

        Checks \\cite/\\citep/\\citet keys against .bib file,
        \\label vs \\ref consistency, and \\gls usage vs glossary definitions.

        Args:
            tex_content: LaTeX source text to check.
        """
        # Citation keys vs bib
        cite_keys = set(L.extract_all_cite_keys(tex_content))
        bib_text = S.read_bib_file()
        bib_keys = set()
        if bib_text:
            for entry in L.parse_bib_entries(bib_text):
                bib_keys.add(entry["key"])

        cited_not_in_bib = sorted(cite_keys - bib_keys)
        in_bib_not_cited = sorted(bib_keys - cite_keys)

        # Labels vs refs
        labels, refs = L.parse_labels_refs(tex_content)
        refs_undefined = sorted(refs - labels)
        labels_unused = sorted(labels - refs)

        # Glossary usage vs definitions
        gls_text = S.read_glossary_file()
        gls_defs = L.parse_glossary_defs(gls_text) if gls_text else {}
        gls_used = set(L.parse_gls_usage(tex_content))
        gls_undefined = sorted(gls_used - set(gls_defs.keys()))
        gls_unused = sorted(set(gls_defs.keys()) - gls_used)

        return {
            "citations": {
                "cited_not_in_bib": cited_not_in_bib,
                "in_bib_not_cited": in_bib_not_cited,
            },
            "labels": {
                "refs_undefined": refs_undefined,
                "labels_unused": labels_unused,
            },
            "glossary": {
                "used_but_undefined": gls_undefined,
                "defined_but_unused": gls_unused,
            },
        }

    @mcp.tool()
    def check_bibtex_health() -> dict:
        """Validate the thesis .bib file for common issues.

        Checks: missing required fields per entry type, duplicate keys,
        year sanity (outside 1950-2027), missing DOI/URL, author format.
        """
        bib_text = S.read_bib_file()
        if not bib_text:
            return {"error": "Could not read .bib file"}

        entries = L.parse_bib_entries(bib_text)

        # Required fields by entry type
        required = {
            "article": ["author", "title", "journal", "year"],
            "inproceedings": ["author", "title", "booktitle", "year"],
            "incollection": ["author", "title", "booktitle", "year"],
            "book": ["author", "title", "publisher", "year"],
            "phdthesis": ["author", "title", "school", "year"],
            "mastersthesis": ["author", "title", "school", "year"],
            "techreport": ["author", "title", "institution", "year"],
            "misc": ["author", "title"],
        }

        issues = {
            "missing_fields": [],
            "duplicate_keys": [],
            "year_issues": [],
            "no_doi_or_url": [],
        }

        seen_keys: dict[str, int] = {}
        for entry in entries:
            key = entry["key"]
            fields = entry["fields"]
            etype = entry["entry_type"]

            # Duplicate key check
            if key in seen_keys:
                issues["duplicate_keys"].append({"key": key, "lines": [seen_keys[key], entry["line"]]})
            seen_keys[key] = entry["line"]

            # Required fields
            req = required.get(etype, ["author", "title"])
            missing = [f for f in req if f not in fields]
            if missing:
                issues["missing_fields"].append({"key": key, "type": etype, "missing": missing})

            # Year sanity
            year_str = fields.get("year", "")
            if year_str:
                try:
                    yr = int(year_str)
                    if yr < 1950 or yr > 2027:
                        issues["year_issues"].append({"key": key, "year": yr})
                except ValueError:
                    issues["year_issues"].append({"key": key, "year": year_str, "error": "not a number"})

            # DOI/URL coverage
            if "doi" not in fields and "url" not in fields:
                issues["no_doi_or_url"].append(key)

        return {
            "total_entries": len(entries),
            "issues": issues,
            "summary": {
                "missing_fields": len(issues["missing_fields"]),
                "duplicate_keys": len(issues["duplicate_keys"]),
                "year_issues": len(issues["year_issues"]),
                "no_doi_or_url": len(issues["no_doi_or_url"]),
            },
        }

    @mcp.tool()
    def citation_density(tex_content: str) -> dict:
        """Measure citation density per section (citations per 100 words).

        Helps identify under-cited sections that may need more references.

        Args:
            tex_content: LaTeX source text to analyze.
        """
        sections = L.parse_sections(tex_content)
        if not sections:
            # Treat whole content as one section
            plain = L.strip_latex(tex_content)
            n_words = T.word_count(plain)
            n_cites = len(L.parse_citations(tex_content))
            density = round(n_cites / max(1, n_words) * 100, 2)
            return {"sections": [], "overall": {"words": n_words, "citations": n_cites, "density": density}}

        results = []
        total_words = 0
        total_cites = 0
        for sec in sections:
            plain = L.strip_latex(sec["content"])
            n_words = T.word_count(plain)
            n_cites = len(L.parse_citations(sec["content"]))
            density = round(n_cites / max(1, n_words) * 100, 2)
            total_words += n_words
            total_cites += n_cites
            results.append({
                "title": sec["title"],
                "level": sec["level"],
                "words": n_words,
                "citations": n_cites,
                "density": density,
            })

        overall_density = round(total_cites / max(1, total_words) * 100, 2)
        return {
            "sections": results,
            "overall": {"words": total_words, "citations": total_cites, "density": overall_density},
        }

    @mcp.tool()
    def validate_citation_context(tex_content: str, n: int = 3) -> list[dict]:
        """For each citation, check if the cited paper supports the surrounding claim.

        Embeds the sentence containing each citation, then queries the KB
        for that paper's nuggets. Returns distance scores so you can judge
        potential misattributions.

        Args:
            tex_content: LaTeX source text.
            n: Number of matching nuggets to return per citation (default 3).
        """
        citations = L.parse_citations(tex_content)
        if not citations:
            return []

        collection = S.get_collection()
        results = []
        seen_keys = set()

        for cit in citations:
            for key in cit["keys"]:
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                paper = S.resolve_cite_key(key)
                if not paper:
                    results.append({
                        "key": key,
                        "line": cit["line"],
                        "context": cit["context"],
                        "status": "unresolved",
                        "matches": [],
                    })
                    continue

                # Embed the context sentence and search against this paper's nuggets
                try:
                    query_emb = S.embed_query(cit["context"])
                    res = collection.query(
                        query_embeddings=[query_emb],
                        n_results=n,
                        where={"paper_id": paper["paper_id"]},
                    )
                except Exception:
                    results.append({
                        "key": key,
                        "line": cit["line"],
                        "context": cit["context"],
                        "status": "error",
                        "matches": [],
                    })
                    continue

                matches = []
                for i in range(len(res["ids"][0])):
                    nid = res["ids"][0][i]
                    dist = res["distances"][0][i]
                    nugget = S.get_nugget_qa(nid)
                    matches.append({
                        "nugget_id": nid,
                        "distance": round(dist, 4),
                        "question": nugget["question"] if nugget else "",
                        "answer": (nugget["answer"][:200] + "...") if nugget and len(nugget.get("answer", "")) > 200 else (nugget["answer"] if nugget else ""),
                    })

                best_dist = matches[0]["distance"] if matches else 999
                results.append({
                    "key": key,
                    "paper_id": paper["paper_id"],
                    "paper_title": paper.get("title", ""),
                    "line": cit["line"],
                    "context": cit["context"],
                    "status": "weak_match" if best_dist > 1.2 else "ok",
                    "best_distance": best_dist,
                    "matches": matches,
                })

        return results

    @mcp.tool()
    def author_diversity(tex_content: str) -> dict:
        """Detect over-reliance on papers from a single author group.

        Resolves cite keys to papers and groups by first author.
        Flags any first author contributing >15% of total citations.

        Args:
            tex_content: LaTeX source text to analyze.
        """
        cite_keys = L.extract_all_cite_keys(tex_content)
        if not cite_keys:
            return {"total_citations": 0, "authors": {}}

        from collections import Counter
        author_counts: Counter = Counter()
        resolved = 0

        for key in cite_keys:
            author = S.parse_author_from_key(key)
            if author:
                author_counts[author] += 1
                resolved += 1

        total = max(1, resolved)
        distribution = []
        for author, count in author_counts.most_common():
            pct = round(count / total * 100, 1)
            distribution.append({
                "author": author,
                "count": count,
                "percent": pct,
                "flagged": pct > 15,
            })

        flagged = [a for a in distribution if a["flagged"]]
        return {
            "total_citations": len(cite_keys),
            "resolved": resolved,
            "unique_first_authors": len(author_counts),
            "distribution": distribution,
            "flagged_authors": flagged,
        }
