"""Parse Zotero pipe-delimited export into lookup table."""
import re
from difflib import SequenceMatcher


def _norm(title):
    t = title.lower().strip()
    t = re.sub(r"[^a-z0-9\s\-]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def parse_zotero_export(filepath):
    """Parse pipe-delimited: internal_id|arxiv_id|title|year"""
    entries = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 4:
                continue
            arxiv_id = parts[1].strip() or None
            title = parts[2].strip()
            try:
                year = int(parts[3].strip())
            except ValueError:
                year = None
            entries.append({
                "zotero_id": parts[0].strip(),
                "arxiv_id": arxiv_id,
                "title": title,
                "year": year,
                "title_norm": _norm(title),
            })
    return entries


def match_pdf_to_zotero(pdf_entry, zotero_entries, threshold=0.75):
    """Find best Zotero match for a parsed PDF entry."""
    pdf_norm = pdf_entry.get("title_normalized", "")
    if not pdf_norm or len(pdf_norm) < 5:
        return None
    best, best_s = None, 0.0
    for ze in zotero_entries:
        s = SequenceMatcher(None, pdf_norm, ze["title_norm"]).ratio()
        if s > best_s:
            best_s = s
            best = ze
    return best if best_s >= threshold else None
