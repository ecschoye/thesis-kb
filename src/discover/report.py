"""Output generation — JSON report + terminal summary."""

import json
import os
from datetime import datetime


def generate_report(scored, dedup_stats, total_raw, sources_used, output_dir=None):
    """Write JSON report and return terminal summary string."""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")

    recommendations = []
    for i, item in enumerate(scored):
        p = item["paper"]
        recommendations.append({
            "rank": i + 1,
            "title": p.title,
            "authors": p.authors[:5],  # cap for readability
            "year": p.year,
            "abstract": p.abstract[:500] if p.abstract else "",
            "relevance_score": item["relevance"],
            "embedding_sim": item["embedding_sim"],
            "keyword_score": item["keyword_score"],
            "authority_score": item["authority_score"],
            "arxiv_id": p.arxiv_id,
            "doi": p.doi,
            "source": p.source,
            "citation_count": p.citation_count,
            "url": p.url,
            "pdf_url": p.pdf_url,
        })

    report = {
        "run_date": now.isoformat(),
        "sources_queried": sources_used,
        "total_raw_candidates": total_raw,
        "dedup_stats": dedup_stats,
        "after_dedup": dedup_stats.get("kept", 0),
        "recommendations_count": len(recommendations),
        "recommendations": recommendations,
    }

    # Save JSON
    report_path = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, f"{date_str}.json")
        tmp = report_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        os.replace(tmp, report_path)

    return report, report_path


def format_terminal_summary(report, report_path=None, max_show=20):
    """Format a readable terminal summary."""
    lines = []
    date = report["run_date"][:10]
    lines.append(f"\n{'='*60}")
    lines.append(f"  Paper Discovery Report ({date})")
    lines.append(f"{'='*60}")

    sources = report["sources_queried"]
    lines.append(f"  Sources: {', '.join(sources)}")
    lines.append(
        f"  Raw candidates: {report['total_raw_candidates']} → "
        f"{report['after_dedup']} new after dedup"
    )

    ds = report["dedup_stats"]
    lines.append(
        f"  Filtered: {ds.get('exact_id',0)} by ID, "
        f"{ds.get('fuzzy_title',0)} by title, "
        f"{ds.get('cross_source',0)} cross-source dups"
    )

    recs = report["recommendations"][:max_show]
    if recs:
        lines.append(f"\n  Top {len(recs)} Recommendations:")
        lines.append(f"  {'-'*56}")
        for r in recs:
            score = r["relevance_score"]
            title = r["title"][:70]
            year = r["year"] or "?"
            cites = r["citation_count"]
            src = r["source"]
            aid = r["arxiv_id"]

            authors = r["authors"]
            author_str = authors[0].split()[-1] if authors else "?"
            if len(authors) > 1:
                author_str += " et al."

            lines.append(f"  {r['rank']:>3}. [{score:.2f}] {title}")
            lines.append(
                f"       {author_str} ({year}) | {src} | "
                f"cites: {cites}"
                + (f" | arXiv:{aid}" if aid else "")
            )
    else:
        lines.append("\n  No new papers found.")

    if report_path:
        lines.append(f"\n  Full report: {report_path}")
    lines.append("")
    return "\n".join(lines)
