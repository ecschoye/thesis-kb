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


def detect_section(text_block):
    """Check if a text block is a section heading."""
    first_line = text_block.strip().split("\n")[0].lower()[:80]
    for pattern, name in SECTION_PATTERNS:
        if re.match(pattern, first_line, re.IGNORECASE):
            return name
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
        section = detect_section(text)
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

        pdf_path = paper["local_pdf"]
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
