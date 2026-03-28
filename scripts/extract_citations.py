#!/usr/bin/env python3
"""Extract all citations from PDFs and output a deduplicated CSV."""
import argparse
import csv
import os
import re
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.extract.pdf_to_text import extract_pdf


# ---------------------------------------------------------------------------
# 1. Extract references text from a PDF
# ---------------------------------------------------------------------------

_REF_HEADING_RE = re.compile(
    r"^\s*(?:references|bibliography)\s*$", re.IGNORECASE | re.MULTILINE
)
_BRACKET_CITE_RE = re.compile(r"\[\d+\]\s+[A-Z]")


def extract_references_text(pdf_path):
    """Return the raw text of the references section, or None."""
    result = extract_pdf(pdf_path)

    # 1. Try section-detected references
    ref_pages = [p for p in result["pages"] if p["section"] == "references"]
    if ref_pages:
        return "\n".join(p["text"] for p in ref_pages)

    # 2. Fallback: scan last 6 pages for a "References" heading or dense [N] patterns
    pages = result["pages"]
    if not pages:
        return None

    tail = pages[-min(6, len(pages)):]
    for i, page in enumerate(tail):
        text = page["text"]
        # Check for explicit heading
        heading_m = _REF_HEADING_RE.search(text)
        if heading_m:
            # Take text from heading onward + remaining pages
            ref_text = text[heading_m.start():]
            for later in tail[i + 1:]:
                ref_text += "\n" + later["text"]
            return ref_text
        # Check for dense bracket citations (>=5 on one page)
        if len(_BRACKET_CITE_RE.findall(text)) >= 5:
            ref_text = text
            for later in tail[i + 1:]:
                ref_text += "\n" + later["text"]
            return ref_text

    return None


# ---------------------------------------------------------------------------
# 2. Split references block into individual entries
# ---------------------------------------------------------------------------

_BRACKET_RE = re.compile(r"\n\s*\[(\d+)\]")
_DOT_NUM_RE = re.compile(r"\n\s*(\d+)\.\s+[A-Z]")


def split_references(text):
    """Split a references block into individual entry strings."""
    # Strip everything before the first reference marker
    # Auto-detect format: bracket [N] vs numbered N.
    bracket_hits = _BRACKET_RE.findall(text)
    dot_hits = _DOT_NUM_RE.findall(text)

    if len(bracket_hits) >= 3:
        parts = _BRACKET_RE.split(text)
        # parts = [preamble, num1, body1, num2, body2, ...]
        entries = []
        for i in range(1, len(parts) - 1, 2):
            body = parts[i + 1].strip() if i + 1 < len(parts) else ""
            body = _clean_entry(body)
            if len(body) > 20:
                entries.append(body)
        return entries

    if len(dot_hits) >= 3:
        parts = _DOT_NUM_RE.split(text)
        entries = []
        for i in range(1, len(parts) - 1, 2):
            body = parts[i + 1].strip() if i + 1 < len(parts) else ""
            body = _clean_entry(body)
            if len(body) > 20:
                entries.append(body)
        return entries

    # Fallback: split on blank lines
    entries = []
    for block in re.split(r"\n\s*\n", text):
        block = _clean_entry(block)
        if len(block) > 20:
            entries.append(block)
    return entries


def _clean_entry(s):
    """Collapse whitespace, rejoin hyphenated line breaks."""
    s = re.sub(r"(\w)-\n(\w)", r"\1\2", s)  # rejoin hyphenation
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------------------------------------------------------------------------
# 3. Parse structured fields from a single reference entry
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(r"\b(19[89]\d|20[0-2]\d)\b")
_ARXIV_RE = re.compile(r"arXiv[:\s]*(\d{4})\.\d{4,5}", re.IGNORECASE)
_QUOTED_TITLE_RE = re.compile(r'["\u201c](.{10,}?)["\u201d]')


def parse_reference(entry):
    """Parse an entry string into {authors, title, year, venue}."""
    # --- Year ---
    year = None
    arxiv_m = _ARXIV_RE.search(entry)
    year_matches = _YEAR_RE.findall(entry)
    if year_matches:
        year = year_matches[-1]  # last year is usually the pub year
    elif arxiv_m:
        prefix = arxiv_m.group(1)
        year = f"20{prefix[:2]}"

    # --- Title ---
    title = None
    authors = None
    venue = None

    # Try quoted title first (IEEE / journal style)
    qt = _QUOTED_TITLE_RE.search(entry)
    if qt:
        title = qt.group(1).strip().rstrip(",.")
        authors = entry[:qt.start()].strip().rstrip(",.")
        after_title = entry[qt.end():].strip().lstrip(",. ")
        venue = _extract_venue(after_title, year)
    else:
        # Period-delimited: Authors. Title. Venue, Year.
        # Split on periods that are followed by a space and uppercase
        # but skip initials like "A. B."
        segments = _split_on_title_boundary(entry)
        if len(segments) >= 2:
            authors = segments[0].strip().rstrip(",.")
            title = segments[1].strip().rstrip(",.")
            rest = ". ".join(segments[2:]) if len(segments) > 2 else ""
            venue = _extract_venue(rest, year)
        else:
            # Can't parse — put everything in title
            title = entry[:200]

    # Clean up authors
    if authors:
        # Remove trailing "and" artifacts
        authors = re.sub(r",?\s*$", "", authors).strip()

    # Post-parse: if "title" looks like a venue fragment, swap title/authors
    if title and _looks_like_venue(title) and authors:
        # The real title is probably in the authors field
        # This happens when period-split lands on a venue abbreviation
        title = authors
        authors = ""

    # Cap title length — anything over 300 chars is a parsing failure
    if title and len(title) > 300:
        title = title[:300]

    return {
        "authors": authors or "",
        "title": title or "",
        "year": year or "",
        "venue": venue or "",
    }


def _split_on_title_boundary(entry):
    """Split entry into [authors, title, rest...] using period boundaries."""
    # Match periods that end a segment (not initials like "A." or "Jr.")
    # A title boundary period: preceded by >=3 word chars, followed by space+uppercase or quote
    parts = re.split(r'(?<=[a-z\d]{2})\.\s+(?=[A-Z"\u201c])', entry, maxsplit=2)
    if len(parts) >= 2:
        return parts
    # Fallback: split on first comma-space after author block
    # Look for pattern: "Lastname, F., Lastname, F., ... Title..."
    # This is too ambiguous; just return as-is
    return [entry]


_VENUE_FRAGS = re.compile(
    r"^(In\s+)?(IEEE|ACM|CVPR|ICCV|ECCV|NeurIPS|ICML|AAAI|Proc\.|Conf\.|Trans\.|"
    r"Int\.\s*Conf|Eur\.\s*Conf|arXiv|Springer|LNCS|Frontiers)",
    re.IGNORECASE,
)


def _looks_like_venue(text):
    """Return True if text looks like a venue name rather than a paper title."""
    return bool(_VENUE_FRAGS.match(text.strip())) and len(text.strip()) < 60


def _extract_venue(text, year):
    """Extract venue from text after the title, stripping year/pages."""
    if not text:
        return ""
    # Remove year and page numbers
    text = re.sub(r"\b(19[89]\d|20[0-2]\d)\b", "", text)
    text = re.sub(r"\bpp?\.\s*\d+[\d\s,–-]*", "", text)
    text = re.sub(r"\bvol\.\s*\d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bno\.\s*\d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\d+\s*$", "", text)  # trailing numbers
    # Remove trailing punctuation / reference backrefs like "1, 2, 5"
    text = re.sub(r"[\s,.\d]+$", "", text)
    text = text.strip(" ,.")
    # Truncate if too long (likely parsing error)
    if len(text) > 200:
        text = text[:200]
    return text


# ---------------------------------------------------------------------------
# 4. Deduplication
# ---------------------------------------------------------------------------

def _normalize_title(title):
    """Lowercase, strip punctuation, collapse whitespace."""
    t = title.lower()
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def deduplicate(entries):
    """Deduplicate by normalized title. Returns list of unique entries
    with occurrence_count and source_pdfs fields."""
    seen = {}  # normalized_title -> entry
    for e in entries:
        key = _normalize_title(e["title"])
        if not key or len(key) < 10:
            continue
        if key in seen:
            seen[key]["occurrence_count"] += 1
            seen[key]["source_pdfs"].add(e["source_pdf"])
            # Prefer the entry with more complete fields
            for field in ("authors", "year", "venue"):
                if not seen[key][field] and e.get(field):
                    seen[key][field] = e[field]
        else:
            seen[key] = {
                **e,
                "occurrence_count": 1,
                "source_pdfs": {e["source_pdf"]},
            }
    return sorted(seen.values(), key=lambda x: -x["occurrence_count"])


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Extract citations from PDFs")
    ap.add_argument("--input-dir", default=os.path.expanduser("~/Documents/thesis-papers"),
                    help="Directory with PDFs")
    ap.add_argument("--output", default="citations.csv",
                    help="Output CSV path")
    ap.add_argument("--raw-output", default=None,
                    help="Optional: also write per-PDF raw entries CSV")
    args = ap.parse_args()

    pdf_dir = Path(args.input_dir)
    pdfs = sorted(f for f in pdf_dir.iterdir() if f.suffix.lower() == ".pdf")
    print(f"Found {len(pdfs)} PDFs in {pdf_dir}")

    all_entries = []
    stats = {"processed": 0, "no_refs": 0, "errors": 0, "total_refs": 0}

    for pdf_path in tqdm(pdfs, desc="Extracting references"):
        try:
            text = extract_references_text(str(pdf_path))
        except Exception:
            stats["errors"] += 1
            continue

        if not text:
            stats["no_refs"] += 1
            continue

        stats["processed"] += 1
        raw_entries = split_references(text)

        for entry_text in raw_entries:
            parsed = parse_reference(entry_text)
            parsed["source_pdf"] = pdf_path.stem
            parsed["raw_text"] = entry_text[:500]
            all_entries.append(parsed)
            stats["total_refs"] += 1

    print("\nExtraction complete:")
    print(f"  PDFs processed:  {stats['processed']}")
    print(f"  No refs section: {stats['no_refs']}")
    print(f"  Errors:          {stats['errors']}")
    print(f"  Total refs:      {stats['total_refs']}")

    # Write raw per-PDF entries if requested
    if args.raw_output:
        _write_csv(args.raw_output, all_entries,
                   ["source_pdf", "authors", "title", "year", "venue", "raw_text"])
        print(f"  Raw CSV:         {args.raw_output}")

    # Deduplicate
    unique = deduplicate(all_entries)
    # Convert source_pdfs set to semicolon-separated string
    for e in unique:
        e["source_pdfs"] = ";".join(sorted(e["source_pdfs"]))

    out_path = args.output
    _write_csv(out_path, unique,
               ["title", "authors", "year", "venue", "occurrence_count", "source_pdfs"])
    print(f"  Unique refs:     {len(unique)}")
    print(f"  Output:          {out_path}")


def _write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
