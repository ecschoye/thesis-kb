"""Extract text from PDFs using PyMuPDF with section detection."""
import os, re, argparse, json
import fitz  # PyMuPDF
from pathlib import Path
from src.utils import load_config, load_json, save_json, already_processed


SECTION_PATTERNS = [
    (r"^\s*abstract\b", "abstract"),
    (r"^\s*(?:[\d.]+\s+)?introduction\b", "introduction"),
    (r"^\s*(?:[\d.]+\s+)?related\s+work", "related_work"),
    (r"^\s*(?:[\d.]+\s+)?background\b", "background"),
    (r"^\s*(?:[\d.]+\s+)?(?:method|approach|proposed)", "method"),
    (r"^\s*(?:[\d.]+\s+)?(?:experiment|evaluation)", "experiments"),
    (r"^\s*(?:[\d.]+\s+)?(?:result|finding)", "results"),
    (r"^\s*(?:[\d.]+\s+)?discussion\b", "discussion"),
    (r"^\s*(?:[\d.]+\s+)?(?:conclusion|summary)", "conclusion"),
    (r"^\s*(?:[\d.]+\s+)?(?:acknowledg)", "acknowledgments"),
    (r"^\s*(?:references|bibliography)\b", "references"),
    (r"^\s*(?:[A-Z]\.?\s+)?(?:appendix|supplementary)", "appendix"),
]


def _detect_section_from_lines(lines):
    """Check if any of the given lines match a section heading pattern."""
    for line in lines:
        line = line.strip().lower()[:80]
        if not line:
            continue
        for pattern, name in SECTION_PATTERNS:
            if re.match(pattern, line, re.IGNORECASE):
                return name
    return None


def _detect_section_from_fonts(page):
    """Detect section headings by finding large-font text blocks."""
    try:
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
    except Exception:
        return None

    # Collect font sizes from text spans
    font_sizes = []
    text_spans = []
    for block in blocks:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                size = span.get("size", 0)
                if text and size > 0:
                    font_sizes.append(size)
                    text_spans.append((size, text))

    if len(font_sizes) < 3:
        return None

    # Find the median font size (body text) and look for larger text (headings)
    sorted_sizes = sorted(font_sizes)
    median_size = sorted_sizes[len(sorted_sizes) // 2]
    heading_threshold = median_size * 1.15  # headings are typically ≥15% larger

    for size, text in text_spans:
        if size >= heading_threshold:
            line_lower = text.lower()[:80]
            for pattern, name in SECTION_PATTERNS:
                if re.match(pattern, line_lower, re.IGNORECASE):
                    return name
    return None


# Roman numeral section heading patterns (e.g. "I. Introduction", "IV. Results")
ROMAN_SECTION_PATTERNS = [
    (r"^\s*I{1,3}V?\s*[.:]\s*introduction", "introduction"),
    (r"^\s*I{1,3}V?\s*[.:]\s*related\s+work", "related_work"),
    (r"^\s*I{1,3}V?\s*[.:]\s*background", "background"),
    (r"^\s*I{1,3}V?\s*[.:]\s*(?:method|approach|proposed)", "method"),
    (r"^\s*I{1,3}V?\s*[.:]\s*(?:experiment|evaluation)", "experiments"),
    (r"^\s*I{1,3}V?\s*[.:]\s*(?:result|finding)", "results"),
    (r"^\s*I{1,3}V?\s*[.:]\s*discussion", "discussion"),
    (r"^\s*I{1,3}V?\s*[.:]\s*(?:conclusion|summary)", "conclusion"),
    (r"^\s*V?I{0,3}\s*[.:]\s*(?:acknowledg)", "acknowledgments"),
    (r"^\s*(?:references|bibliography)\b", "references"),
    (r"^\s*(?:appendix|supplementary)", "appendix"),
]


def detect_section(text_block, page=None):
    """Check if a text block contains a section heading.

    Scans the first 8 non-empty lines (not just the first line) and
    optionally uses font-size detection via PyMuPDF dict output.
    """
    lines = text_block.strip().split("\n")
    # Take first 8 non-empty lines
    candidate_lines = [l for l in lines if l.strip()][:8]

    # 1. Try standard patterns on first 8 lines
    result = _detect_section_from_lines(candidate_lines)
    if result:
        return result

    # 2. Try Roman numeral patterns
    for line in candidate_lines:
        line_lower = line.strip().lower()[:80]
        if not line_lower:
            continue
        for pattern, name in ROMAN_SECTION_PATTERNS:
            if re.match(pattern, line_lower, re.IGNORECASE):
                return name

    # 3. Try font-size-based detection if page object is available
    if page is not None:
        result = _detect_section_from_fonts(page)
        if result:
            return result

    return None


def extract_page_text(page, page_num=None):
    """Extract text from a single PDF page.

    Returns extracted text, or empty string for scanned/image-only pages
    (OCR is not supported — these pages are skipped with a warning).
    """
    text = page.get_text("text")
    if text and len(text.strip()) > 0:
        return text
    # Page has no extractable text (likely scanned/image-only)
    if page_num is not None:
        print(f"  WARNING: page {page_num} has no extractable text (scanned/image-only)")
    return ""


def extract_pdf(pdf_path, min_text_length=100):
    """Extract text from a PDF, returning structured page data.

    Returns dict with:
        pages: list of {page_num, text, section, char_count}
        total_chars: int
        total_pages: int
        skipped_pages: int
    """
    doc = fitz.open(pdf_path)
    pages = []
    current_section = "preamble"
    skipped = 0

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = extract_page_text(page, page_num=page_num + 1)
        char_count = len(text.strip())

        if char_count < min_text_length:
            skipped += 1
            continue

        # Detect section transitions
        section = detect_section(text, page=page)
        if section:
            current_section = section

        pages.append({
            "page_num": page_num + 1,
            "text": text.strip(),
            "section": current_section,
            "char_count": char_count,
        })

    total_pages = len(doc)
    doc.close()
    total_chars = sum(p["char_count"] for p in pages)
    return {
        "pages": pages,
        "total_chars": total_chars,
        "total_pages": total_pages,
        "skipped_pages": skipped,
    }


def run_extraction(config_path="config.yaml"):
    """Extract text from all PDFs in the corpus."""
    cfg = load_config(config_path)
    pdf_dir = cfg["paths"]["pdf_dir"]
    text_dir = cfg["paths"]["text_dir"]
    corpus_dir = cfg["paths"]["corpus_dir"]
    min_len = cfg.get("extract", {}).get("min_text_length", 100)
    os.makedirs(text_dir, exist_ok=True)

    # Load manifest for paper IDs
    manifest_path = os.path.join(corpus_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        print("No manifest.json found. Run ingest first.")
        return
    manifest = load_json(manifest_path)

    print(f"[extract] Processing {len(manifest)} papers...")
    success, failed, skipped_count = 0, 0, 0

    for i, paper in enumerate(manifest):
        paper_id = paper["paper_id"]
        if already_processed(paper_id, text_dir):
            skipped_count += 1
            continue

        # Use pdf_dir from config + filename, since local_pdf may have a stale absolute path
        pdf_filename = os.path.basename(paper["local_pdf"])
        pdf_path = os.path.join(pdf_dir, pdf_filename)
        if not os.path.exists(pdf_path):
            print(f"  [{i+1}] SKIP (file missing): {pdf_path}")
            failed += 1
            continue

        try:
            result = extract_pdf(pdf_path, min_text_length=min_len)
            result["paper_id"] = paper_id
            result["title"] = paper.get("title", "")
            result["source_pdf"] = pdf_path
            save_json(result, os.path.join(text_dir, f"{paper_id}.json"))
            success += 1
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(manifest)}] {success} ok, {failed} err, {skipped_count} skip")
        except Exception as e:
            print(f"  [{i+1}] ERROR {paper_id}: {e}")
            failed += 1

    print(f"\nDone: {success} extracted, {failed} failed, {skipped_count} skipped")


def main():
    ap = argparse.ArgumentParser(description="Extract text from PDFs")
    ap.add_argument("-c", "--config", default="config.yaml")
    args = ap.parse_args()
    run_extraction(args.config)


if __name__ == "__main__":
    main()
