"""Extract text from PDFs using PyMuPDF with section detection."""
import os
import re
import argparse
import logging
import fitz  # PyMuPDF
from src.utils import load_config, load_json, save_json, already_processed

log = logging.getLogger(__name__)

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


def detect_section(text_block, page=None):
    """Check if a text block contains a section heading.

    Scans the first 8 non-empty lines (not just the first line) and
    optionally uses font-size detection via PyMuPDF dict output.
    """
    lines = text_block.strip().split("\n")
    # Take first 8 non-empty lines
    candidate_lines = [line for line in lines if line.strip()][:8]

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


# ---------------------------------------------------------------------------
# Multi-column detection and reading-order reordering
# ---------------------------------------------------------------------------

def _reorder_blocks_reading_order(page):
    """Extract text blocks and reorder for correct reading order.

    Detects multi-column layouts by clustering block x-positions.
    For two-column pages, reads left column top-to-bottom, then right.
    Single-column pages are returned in natural (y, x) order.

    Returns the full page text as a string.
    """
    raw_blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, type)
    # Filter to text blocks only (type 0), skip images (type 1)
    text_blocks = [b for b in raw_blocks if b[6] == 0 and b[4].strip()]
    if not text_blocks:
        return ""

    page_rect = page.rect
    page_width = page_rect.width
    page_height = page_rect.height

    # Separate header/footer blocks (span >60% of page width, top/bottom 8%)
    header_zone = page_height * 0.08
    footer_zone = page_height * 0.92
    wide_threshold = page_width * 0.60

    headers = []
    footers = []
    body_blocks = []
    for b in text_blocks:
        x0, y0, x1, y1 = b[0], b[1], b[2], b[3]
        block_width = x1 - x0
        if block_width >= wide_threshold:
            if y0 < header_zone:
                headers.append(b)
                continue
            if y1 > footer_zone:
                footers.append(b)
                continue
        body_blocks.append(b)

    if not body_blocks:
        # Page is all headers/footers — just return in order
        all_blocks = sorted(headers + footers, key=lambda b: (b[1], b[0]))
        return "\n".join(b[4].strip() for b in all_blocks)

    # Detect columns by analyzing left-edge x-positions of body blocks
    left_edges = sorted(b[0] for b in body_blocks)
    mid_x = page_width / 2

    # Count blocks clearly on each side (with 15% margin from center)
    margin = page_width * 0.15
    left_count = sum(1 for x in left_edges if x < mid_x - margin)
    right_count = sum(1 for x in left_edges if x > mid_x + margin)

    # Check for a gap in x-positions near the page center
    # Two-column layout: most blocks start either left of center or right of center
    is_two_column = (left_count >= 2 and right_count >= 2)

    if is_two_column:
        # Find the split point: largest gap in the center region
        # Also consider right edges for gap detection
        all_edges = []
        for b in body_blocks:
            all_edges.append(("left", b[0]))
            all_edges.append(("right", b[2]))

        # Simple split: use page midpoint
        split_x = mid_x

        left_col = [b for b in body_blocks if b[0] < split_x]
        right_col = [b for b in body_blocks if b[0] >= split_x]

        # Wide blocks that span both columns go to whichever has more overlap
        # (already handled: they stay in left_col since x0 < split_x)

        left_col.sort(key=lambda b: (b[1], b[0]))
        right_col.sort(key=lambda b: (b[1], b[0]))

        ordered = (
            sorted(headers, key=lambda b: (b[1], b[0]))
            + left_col
            + right_col
            + sorted(footers, key=lambda b: (b[1], b[0]))
        )
    else:
        # Single column: sort by (y, x)
        body_blocks.sort(key=lambda b: (b[1], b[0]))
        ordered = (
            sorted(headers, key=lambda b: (b[1], b[0]))
            + body_blocks
            + sorted(footers, key=lambda b: (b[1], b[0]))
        )

    return "\n".join(b[4].strip() for b in ordered)


# ---------------------------------------------------------------------------
# OCR support
# ---------------------------------------------------------------------------

_ocr_available = None  # None = not tested yet, True/False after first attempt


def _check_ocr_available():
    """Test whether Tesseract OCR is available via PyMuPDF."""
    global _ocr_available
    if _ocr_available is not None:
        return _ocr_available
    try:
        # Create a tiny test page to see if OCR works
        test_doc = fitz.open()
        test_page = test_doc.new_page(width=100, height=100)
        test_page.get_textpage_ocr(flags=0, dpi=72, full=True)
        test_doc.close()
        _ocr_available = True
    except Exception as e:
        _ocr_available = False
        log.warning("Tesseract OCR not available: %s. OCR fallback disabled.", e)
    return _ocr_available


def _ocr_page(page, dpi=300):
    """Run OCR on a page and return extracted text."""
    tp = page.get_textpage_ocr(
        flags=fitz.TEXT_PRESERVE_WHITESPACE,
        dpi=dpi,
        full=True,
    )
    return page.get_text("text", textpage=tp)


# ---------------------------------------------------------------------------
# Page and PDF extraction
# ---------------------------------------------------------------------------

def extract_page_text(page, page_num=None, ocr_fallback=False, column_detection=True,
                      min_chars=50):
    """Extract text from a single PDF page.

    Returns (text, used_ocr) tuple.
    - column_detection: use block-level extraction with reading-order reordering
    - ocr_fallback: attempt OCR on pages with insufficient text
    """
    if column_detection:
        text = _reorder_blocks_reading_order(page)
    else:
        text = page.get_text("text")

    if text and len(text.strip()) >= min_chars:
        return text, False

    # Page has no/insufficient extractable text — try OCR
    if ocr_fallback and _check_ocr_available():
        try:
            ocr_text = _ocr_page(page)
            if ocr_text and len(ocr_text.strip()) >= min_chars:
                if page_num is not None:
                    log.info("OCR recovered %d chars on page %d", len(ocr_text.strip()), page_num)
                return ocr_text, True
        except Exception as e:
            if page_num is not None:
                log.warning("OCR failed on page %d: %s", page_num, e)

    if page_num is not None and (not text or len(text.strip()) < min_chars):
        print(f"  WARNING: page {page_num} has no extractable text (scanned/image-only)")
    return text or "", False


def extract_pdf(pdf_path, min_text_length=100, ocr_fallback=False, column_detection=True):
    """Extract text from a PDF, returning structured page data.

    Returns dict with:
        pages: list of {page_num, text, section, char_count}
        total_chars: int
        total_pages: int
        skipped_pages: int
        ocr_pages: int
        columns_detected: bool (True if any page used column reordering)
    """
    doc = fitz.open(pdf_path)
    pages = []
    current_section = "preamble"
    skipped = 0
    ocr_pages = 0

    for page_num in range(len(doc)):
        page = doc[page_num]
        text, used_ocr = extract_page_text(
            page,
            page_num=page_num + 1,
            ocr_fallback=ocr_fallback,
            column_detection=column_detection,
            min_chars=min_text_length,
        )
        if used_ocr:
            ocr_pages += 1

        # Always check for section transitions, even on sparse pages
        section = detect_section(text, page=page)
        if section:
            current_section = section

        char_count = len(text.strip())
        if char_count < min_text_length:
            skipped += 1
            continue

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
        "ocr_pages": ocr_pages,
    }


def run_extraction(config_path="config.yaml"):
    """Extract text from all PDFs in the corpus."""
    cfg = load_config(config_path)
    pdf_dir = cfg["paths"]["pdf_dir"]
    text_dir = cfg["paths"]["text_dir"]
    corpus_dir = cfg["paths"]["corpus_dir"]
    extract_cfg = cfg.get("extract", {})
    min_len = extract_cfg.get("min_text_length", 100)
    ocr_fallback = extract_cfg.get("ocr_fallback", False)
    column_detection = extract_cfg.get("column_detection", True)
    os.makedirs(text_dir, exist_ok=True)

    # Load manifest for paper IDs
    manifest_path = os.path.join(corpus_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        log.error("No manifest.json found. Run ingest first.")
        return
    manifest = load_json(manifest_path)

    print(f"[extract] Processing {len(manifest)} papers...")
    print(f"  column_detection={column_detection}, ocr_fallback={ocr_fallback}")
    success, failed, skipped_count = 0, 0, 0
    total_ocr_pages = 0

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
            result = extract_pdf(
                pdf_path,
                min_text_length=min_len,
                ocr_fallback=ocr_fallback,
                column_detection=column_detection,
            )
            result["paper_id"] = paper_id
            result["title"] = paper.get("title", "")
            result["source_pdf"] = pdf_path
            save_json(result, os.path.join(text_dir, f"{paper_id}.json"))
            total_ocr_pages += result.get("ocr_pages", 0)
            success += 1
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(manifest)}] {success} ok, {failed} err, {skipped_count} skip")
        except Exception as e:
            print(f"  [{i+1}] ERROR {paper_id}: {e}")
            failed += 1

    print(f"\nDone: {success} extracted, {failed} failed, {skipped_count} skipped")
    if total_ocr_pages > 0:
        print(f"  OCR applied to {total_ocr_pages} pages total")


def main():
    ap = argparse.ArgumentParser(description="Extract text from PDFs")
    ap.add_argument("-c", "--config", default="config.yaml")
    args = ap.parse_args()
    run_extraction(args.config)


if __name__ == "__main__":
    main()
