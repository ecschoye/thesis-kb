"""Parse PDF filenames to extract metadata."""
import re
from pathlib import Path


def parse_pdf_filename(filename):
    stem = Path(filename).stem
    result = {"raw_filename": filename, "authors_str": "", "year": None, "title": ""}
    m = re.match(r"^(.+?)\s*-\s*(\d{4})\s*-\s*(.+)$", stem)
    if m:
        result["authors_str"] = m.group(1).strip()
        result["year"] = int(m.group(2))
        result["title"] = m.group(3).strip()
        return result
    m = re.match(r"^(.+?)\s*-\s*([A-Z].{10,})$", stem)
    if m:
        result["authors_str"] = m.group(1).strip()
        result["title"] = m.group(2).strip()
        return result
    title = stem.replace("_", " ").strip()
    result["title"] = title if len(title) > 5 else stem
    return result


def parse_authors_str(authors_str):
    if not authors_str:
        return []
    clean = re.sub(r"\s*et al\.?", "", authors_str)
    parts = re.split(r"\s+and\s+", clean)
    return [p.strip() for p in parts if p.strip()]


def normalize_title(title):
    t = title.lower().strip()
    t = re.sub(r"\.\.\.$$", "", t)
    t = re.sub(r"[^a-z0-9\s\-]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def scan_pdf_directory(pdf_dir):
    pdfs = sorted(Path(pdf_dir).glob("*.pdf"))
    results = []
    for pdf_path in pdfs:
        parsed = parse_pdf_filename(pdf_path.name)
        parsed["pdf_path"] = str(pdf_path)
        parsed["paper_id"] = pdf_path.stem
        parsed["authors"] = parse_authors_str(parsed["authors_str"])
        parsed["title_normalized"] = normalize_title(parsed["title"])
        results.append(parsed)
    return results
