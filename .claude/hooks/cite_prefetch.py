#!/usr/bin/env python3
r"""
UserPromptSubmit hook: pre-fetches paper data for \cite{} keys in the prompt.

Reads hook input JSON from stdin, extracts \cite{AuthorLast_YEAR:...} keys,
queries SQLite for matching papers by author name, fetches their nuggets,
and returns additionalContext so the model doesn't need to run lookups.

Only activates when \cite{} keys are present in the prompt.
"""
import json
import logging
import os
import re
import sqlite3
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# ---------------------------------------------------------------------------
# Inline logger (standalone script, can't import from src)
# ---------------------------------------------------------------------------
_log_dir = Path(__file__).resolve().parent.parent.parent / "logs"
_log_dir.mkdir(exist_ok=True)
log = logging.getLogger("cite_prefetch")
_handler = RotatingFileHandler(_log_dir / "cite_prefetch.log", maxBytes=5_000_000, backupCount=3)
_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
log.addHandler(_handler)
log.setLevel(logging.DEBUG)


def find_kb_db():
    """Find the nuggets.db path by reading config.yaml."""
    for base in [
        os.path.expanduser("~/thesis-kb"),
        "/cluster/work/ecschoye/thesis-kb",
    ]:
        config_path = os.path.join(base, "config.yaml")
        if os.path.exists(config_path):
            kb_dir = "kb"
            with open(config_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("kb_dir:"):
                        kb_dir = line.split(":", 1)[1].strip()
                        break
            db_path = os.path.join(base, kb_dir, "nuggets.db")
            if os.path.exists(db_path):
                log.debug("DB found: %s", db_path)
                return db_path
    log.error("nuggets.db not found in any known path")
    return None


def extract_cite_keys(text):
    r"""Extract \cite{...} keys from text. Returns list of raw key strings."""
    return re.findall(r"\\cite\{([^}]+)\}", text)


def parse_author_from_key(key):
    """Extract author last name from cite key like 'AliAkbarpour_2025:Emerging-Trends-DVS'."""
    base = key.split(":")[0] if ":" in key else key
    parts = base.split("_")
    if parts:
        return parts[0]
    return None


def parse_year_from_key(key):
    """Extract year from cite key like 'AliAkbarpour_2025:Emerging-Trends-DVS'."""
    base = key.split(":")[0] if ":" in key else key
    parts = base.split("_")
    for part in parts:
        if part.isdigit() and len(part) == 4:
            return int(part)
    return None


def parse_slug_from_key(key):
    """Extract slug from cite key like 'AliAkbarpour_2025:Emerging-Trends-DVS' -> 'emerging trends dvs'."""
    if ":" not in key:
        return ""
    slug = key.split(":", 1)[1]
    return slug.lower().replace("-", " ").replace("_", " ")


def slug_title_score(slug, title):
    r"""Score how well a cite key slug matches a paper title. Higher = better.

    Uses recall (slug words found in title) as primary score,
    with a small bonus for precision (fewer extra title words = better match).
    """
    if not slug or not title:
        return 0
    slug_words = set(slug.split())
    title_words = set(re.split(r"\W+", title.lower())) - {"", "a", "an", "the", "of", "for", "and", "in", "on", "with", "to", "from", "by"}
    if not slug_words:
        return 0
    overlap = len(slug_words & title_words)
    recall = overlap / len(slug_words)
    # Precision bonus: prefer titles where slug words are a larger fraction
    precision = overlap / len(title_words) if title_words else 0
    return recall * 0.7 + precision * 0.3


def lookup_papers(db, author, year=None):
    """Find papers by author name (and optionally year) in SQLite."""
    query = "SELECT * FROM papers WHERE authors LIKE ?"
    params = [f"%{author}%"]
    if year:
        query += " AND year = ?"
        params.append(year)
    query += " LIMIT 25"
    rows = db.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def lookup_papers_by_title(db, slug_words):
    """Find papers by title substring match using slug words."""
    if not slug_words:
        return []
    # Use first 3 meaningful words from slug for LIKE query
    words = [w for w in slug_words.split() if len(w) > 2][:3]
    if not words:
        return []
    conditions = [f"title LIKE ?" for _ in words]
    params = [f"%{w}%" for w in words]
    query = f"SELECT * FROM papers WHERE {' AND '.join(conditions)} LIMIT 10"
    rows = db.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def get_nuggets(db, paper_id, limit=30):
    """Get nuggets for a paper_id."""
    rows = db.execute(
        "SELECT nugget_id, question, answer, type, confidence, section "
        "FROM nuggets WHERE paper_id = ? LIMIT ?",
        (paper_id, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def main():
    try:
        hook_input = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError) as e:
        log.error("Failed to parse hook input: %s", e)
        sys.exit(0)

    prompt = hook_input.get("prompt", "")
    if not prompt:
        sys.exit(0)

    cite_keys = extract_cite_keys(prompt)
    if not cite_keys:
        sys.exit(0)

    log.info("Cite keys found: %s", cite_keys)

    db_path = find_kb_db()
    if not db_path:
        sys.exit(0)

    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row

    # Deduplicate by cite key
    seen_keys = {}
    for key in cite_keys:
        if key not in seen_keys:
            author = parse_author_from_key(key)
            year = parse_year_from_key(key)
            slug = parse_slug_from_key(key)
            seen_keys[key] = {"author": author, "year": year, "slug": slug}

    results = {}
    found_count = 0
    missing_count = 0

    for key, info in seen_keys.items():
        author = info["author"]
        if not author:
            log.warning("Could not parse author from key: %s", key)
            continue

        slug = info["slug"]

        # Gather candidates from multiple sources
        papers = lookup_papers(db, author, year=info["year"])
        best_score = max((slug_title_score(slug, p.get("title", "")) for p in papers), default=0) if slug and papers else 0

        # If year-filtered results are poor or empty, also try without year
        if best_score < 0.8 or not papers:
            no_year = lookup_papers(db, author)
            if no_year:
                seen_ids = {p["paper_id"] for p in papers}
                added = [p for p in no_year if p["paper_id"] not in seen_ids]
                if added:
                    log.info("Key %s: added %d candidates from year-relaxed author search", key, len(added))
                    papers.extend(added)

        # Title-based fallback
        if slug and (not papers or best_score < 0.8):
            title_papers = lookup_papers_by_title(db, slug)
            if title_papers:
                seen_ids = {p["paper_id"] for p in papers}
                added = [p for p in title_papers if p["paper_id"] not in seen_ids]
                if added:
                    log.info("Key %s: title fallback added %d candidates for slug '%s'", key, len(added), slug)
                    papers.extend(added)

        if papers:
            if slug and len(papers) > 1:
                scored = [(slug_title_score(slug, p.get("title", "")), p.get("title", "")) for p in papers]
                log.debug("Key %s: slug='%s', scores=%s", key, slug,
                          [(f"{s:.2f}", t[:50]) for s, t in scored])
                papers.sort(
                    key=lambda p: slug_title_score(slug, p.get("title", "")),
                    reverse=True,
                )

            paper = papers[0]
            pid = paper["paper_id"]
            nuggets = get_nuggets(db, pid)
            log.info("Key %s -> paper_id=%s, title='%s', %d nuggets",
                     key, pid, paper.get("title", "")[:60], len(nuggets))
            found_count += 1
            results[key] = {
                "cite_key": key,
                "author_query": author,
                "paper": {
                    "paper_id": pid,
                    "title": paper.get("title", ""),
                    "authors": paper.get("authors", ""),
                    "year": paper.get("year"),
                    "arxiv_id": paper.get("arxiv_id"),
                    "doi": paper.get("doi"),
                },
                "nugget_count": len(nuggets),
                "nuggets": nuggets,
            }
        else:
            log.warning("Key %s: NOT FOUND (author='%s', year=%s)", key, author, info["year"])
            missing_count += 1
            results[key] = {
                "cite_key": key,
                "author_query": author,
                "paper": None,
                "nugget_count": 0,
                "nuggets": [],
            }

    db.close()

    log.info("Done: %d found, %d missing out of %d keys", found_count, missing_count, len(seen_keys))

    if not results:
        sys.exit(0)

    # Build context string
    lines = ["## Pre-fetched Cited Papers (from hook — DO NOT re-query these)"]
    lines.append("")
    # Summary table so model can quickly see found/missing status
    lines.append("### Status Summary")
    for key, data in results.items():
        status = "FOUND" if data["paper"] else "NOT FOUND"
        lines.append(f"- `\\cite{{{key}}}`: **{status}**")
    lines.append("")
    for key, data in results.items():
        lines.append(f"### \\cite{{{key}}}")
        if data["paper"]:
            p = data["paper"]
            arxiv = f" (arXiv:{p['arxiv_id']})" if p.get("arxiv_id") else ""
            lines.append(f"- **Found in KB**: paper_id=`{p['paper_id']}`")
            lines.append(f"- Title: {p['title']}")
            lines.append(f"- Authors: {p['authors']}")
            lines.append(f"- Year: {p['year']}{arxiv}")
            lines.append(f"- Nuggets: {data['nugget_count']}")
            lines.append("")
            for n in data["nuggets"]:
                lines.append(
                    f"  - [{n['type']}] Q: {n['question']}"
                )
                answer = n.get("answer", "")
                if answer:
                    if len(answer) > 300:
                        answer = answer[:300] + "..."
                    lines.append(f"    A: {answer}")
        else:
            lines.append(f"- **Not found in KB** (author query: '{data['author_query']}')")
        lines.append("")

    context = "\n".join(lines)

    output = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": context,
        }
    }
    log.info("Output JSON size: %d bytes", len(json.dumps(output)))
    print(json.dumps(output))
    sys.exit(0)


if __name__ == "__main__":
    main()
