"""Main orchestrator for paper discovery."""

import json
import os
from datetime import date, datetime, timedelta

from src.discover.profile import build_profile
from src.discover.sources import search_all_sources
from src.discover.dedup import deduplicate
from src.discover.scorer import score_candidates, compute_anchor_embeddings
from src.discover.report import generate_report, format_terminal_summary

LEDGER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "corpus", "discovery_ledger.json",
)


def run_discovery(
    config_path="config.yaml",
    bib_path=None,
    max_results=50,
    date_from=None,
    sources=("arxiv", "s2"),
    use_embeddings=True,
    auto_ingest=False,
    ingest_top=10,
    citation_expand_n=20,
    output_dir=None,
    local_papers_dir=None,
):
    """Run the full discovery pipeline.

    Returns the report dict.
    """
    if output_dir is None:
        base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_dir = os.path.join(base, "corpus", "discovery_reports")

    # Default: look back 6 months
    if date_from is None:
        date_from = (datetime.now() - timedelta(days=180)).date()
    elif isinstance(date_from, str):
        date_from = date.fromisoformat(date_from)

    # 1. Build dynamic profile
    print("Building thesis profile from KB...")
    profile = build_profile(config_path, bib_path=bib_path, local_papers_dir=local_papers_dir)
    print(f"  {len(profile.queries)} queries, {len(profile.seed_arxiv_ids)} seed papers")
    print(f"  {len(profile.exclude_arxiv_ids)} papers already in KB")
    print(f"  {len(profile.keyword_terms)} keyword terms extracted")
    if profile.queries:
        print(f"  Sample queries: {profile.queries[:3]}")

    # 2. Search all sources
    print(f"\nSearching sources: {', '.join(sources)}...")
    candidates = search_all_sources(
        queries=profile.queries,
        sources=sources,
        seed_arxiv_ids=profile.seed_arxiv_ids,
        max_per_query=50,
        citation_expand_n=citation_expand_n,
        date_from=date_from,
    )
    total_raw = len(candidates)

    # 3. Deduplicate
    print("\nDeduplicating...")
    new_papers, dedup_stats = deduplicate(candidates, profile)
    print(
        f"  {total_raw} → {len(new_papers)} new "
        f"(removed: {dedup_stats['exact_id']} by ID, "
        f"{dedup_stats['fuzzy_title']} by title, "
        f"{dedup_stats['cross_source']} cross-source)"
    )

    # 4. Score
    print("\nScoring candidates...")
    embed_fn = None
    anchor_embeddings = None

    if use_embeddings:
        try:
            from src.query import ThesisKB
            kb = ThesisKB(config_path)
            embed_fn = kb._embed_query
            anchor_embeddings = compute_anchor_embeddings(profile.queries, embed_fn)
            print(f"  Using embedding scoring ({len(anchor_embeddings)} anchors)")
        except Exception as e:
            print(f"  Embedding scoring unavailable ({e}), using keyword+authority only")
            embed_fn = None
            anchor_embeddings = None

    scored = score_candidates(new_papers, profile, embed_fn, anchor_embeddings)
    top = scored[:max_results]

    # 5. Save ALL scored candidates to persistent ledger
    new_in_ledger = _update_ledger(scored)
    print(f"  Ledger: {new_in_ledger} new entries added ({LEDGER_PATH})")

    # 6. Generate report (top N only)
    report, report_path = generate_report(
        top, dedup_stats, total_raw, list(sources), output_dir
    )
    summary = format_terminal_summary(report, report_path)
    print(summary)

    # 7. Download
    if auto_ingest and top:
        _auto_ingest(top[:ingest_top], config_path)

    return report


def _load_ledger():
    """Load the persistent candidate ledger."""
    if os.path.exists(LEDGER_PATH):
        with open(LEDGER_PATH) as f:
            return json.load(f)
    return {"entries": {}, "stats": {"total_runs": 0, "first_run": None, "last_run": None}}


def _save_ledger(ledger):
    """Save the ledger atomically."""
    os.makedirs(os.path.dirname(LEDGER_PATH), exist_ok=True)
    tmp = LEDGER_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(ledger, f, indent=2, ensure_ascii=False)
    os.replace(tmp, LEDGER_PATH)


def _candidate_key(paper):
    """Stable key for a candidate: prefer arXiv ID, fall back to normalized title."""
    if paper.arxiv_id:
        return f"arxiv:{paper.arxiv_id}"
    if paper.doi:
        return f"doi:{paper.doi.lower()}"
    # Normalize title as fallback key
    import re
    t = paper.title.lower().strip()
    t = re.sub(r"[^a-z0-9\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return f"title:{t}"


def _update_ledger(scored_candidates):
    """Add all scored candidates to the persistent ledger. Returns count of new entries."""
    ledger = _load_ledger()
    now = datetime.now().isoformat()

    ledger["stats"]["total_runs"] = ledger["stats"].get("total_runs", 0) + 1
    if not ledger["stats"].get("first_run"):
        ledger["stats"]["first_run"] = now
    ledger["stats"]["last_run"] = now

    new_count = 0
    for item in scored_candidates:
        p = item["paper"]
        key = _candidate_key(p)

        if key in ledger["entries"]:
            # Update score if this run found a higher one
            existing = ledger["entries"][key]
            if item["relevance"] > existing.get("best_score", 0):
                existing["best_score"] = item["relevance"]
            existing["times_seen"] = existing.get("times_seen", 1) + 1
            existing["last_seen"] = now
        else:
            ledger["entries"][key] = {
                "title": p.title,
                "authors": p.authors[:5],
                "year": p.year,
                "arxiv_id": p.arxiv_id,
                "doi": p.doi,
                "url": p.url,
                "pdf_url": p.pdf_url,
                "citation_count": p.citation_count,
                "best_score": item["relevance"],
                "keyword_score": item["keyword_score"],
                "authority_score": item["authority_score"],
                "embedding_sim": item["embedding_sim"],
                "source": p.source,
                "first_seen": now,
                "last_seen": now,
                "times_seen": 1,
                "downloaded": False,
                "status": "candidate",  # candidate | downloaded | rejected | ingested
            }
            new_count += 1

    _save_ledger(ledger)
    return new_count


def _make_filename(paper):
    """Build 'Author et al. - Year - Title.pdf' matching ~/Documents/thesis-papers convention."""
    authors = paper.authors
    if not authors:
        author_part = "Unknown"
    elif len(authors) == 1:
        author_part = authors[0].split()[-1]  # last name
    else:
        author_part = f"{authors[0].split()[-1]} et al."

    year = paper.year or "Unknown"
    # Clean title for filesystem
    title = paper.title.replace("/", "-").replace(":", " -").replace("\\", "")
    title = " ".join(title.split())[:120]

    return f"{author_part} - {year} - {title}.pdf"


def _auto_ingest(scored_papers, config_path):
    """Download top papers to ~/Documents/thesis-papers."""
    from src.acquire.fetch import download_pdf

    paper_dir = os.path.expanduser("~/Documents/thesis-papers")
    os.makedirs(paper_dir, exist_ok=True)

    downloaded = 0
    skipped = 0
    failed = 0
    print(f"\nDownloading top {len(scored_papers)} papers to {paper_dir}...")
    for item in scored_papers:
        p = item["paper"]
        if not p.pdf_url:
            skipped += 1
            continue

        fname = _make_filename(p)
        dest = os.path.join(paper_dir, fname)

        if os.path.exists(dest):
            skipped += 1
            continue

        ok = download_pdf(p.pdf_url, dest)
        if ok:
            downloaded += 1
            print(f"  [{item['relevance']:.2f}] {p.title[:70]}")
            # Mark in ledger
            _mark_ledger_downloaded(p)
        else:
            failed += 1

    print(f"\n  Done: {downloaded} downloaded, {skipped} skipped, {failed} failed")


def _mark_ledger_downloaded(paper):
    """Mark a paper as downloaded in the ledger."""
    try:
        ledger = _load_ledger()
        key = _candidate_key(paper)
        if key in ledger["entries"]:
            ledger["entries"][key]["downloaded"] = True
            ledger["entries"][key]["status"] = "downloaded"
            _save_ledger(ledger)
    except Exception:
        pass
