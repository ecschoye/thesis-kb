"""Token-based chunking with section awareness."""
import os, argparse
import tiktoken
from src.utils import load_config, load_json, save_json, already_processed


def token_chunks(text, enc, chunk_size=400, overlap=50):
    """Split text into overlapping token windows.

    Returns list of (text, start_tok, end_tok) tuples.
    """
    tokens = enc.encode(text)
    if len(tokens) <= chunk_size:
        return [(text, 0, len(tokens))]
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_text = enc.decode(tokens[start:end])
        chunks.append((chunk_text, start, end))
        if end >= len(tokens):
            break
        start += chunk_size - overlap
    return chunks


def chunk_paper(text_data, enc, chunk_size=400, overlap=50, respect_sections=True):
    """Chunk a paper into overlapping token windows.

    If respect_sections is True, avoids splitting across section
    boundaries by chunking each section independently.
    """
    pages = text_data.get("pages", [])
    if not pages:
        return []

    if respect_sections:
        # Group pages by section
        sections = []
        current_section = None
        current_text = []
        current_pages = []
        for page in pages:
            sec = page.get("section", "preamble")
            if sec != current_section and current_text:
                sections.append({
                    "section": current_section,
                    "text": "\n".join(current_text),
                    "pages": list(current_pages),
                })
                current_text = []
                current_pages = []
            current_section = sec
            current_text.append(page["text"])
            current_pages.append(page["page_num"])
        if current_text:
            sections.append({
                "section": current_section,
                "text": "\n".join(current_text),
                "pages": list(current_pages),
            })

        # Chunk each section independently
        all_chunks = []
        chunk_idx = 0
        for sec in sections:
            raw_chunks = token_chunks(sec["text"], enc, chunk_size, overlap)
            for text, start_tok, end_tok in raw_chunks:
                all_chunks.append({
                    "chunk_id": chunk_idx,
                    "text": text,
                    "section": sec["section"],
                    "pages": sec["pages"],
                    "token_start": start_tok,
                    "token_end": end_tok,
                    "token_count": end_tok - start_tok,
                })
                chunk_idx += 1
        return all_chunks
    else:
        # Simple: concatenate all pages and chunk
        full_text = "\n".join(page["text"] for page in pages)
        raw_chunks = token_chunks(full_text, enc, chunk_size, overlap)
        return [
            {
                "chunk_id": i,
                "text": text,
                "section": "unknown",
                "pages": [pg["page_num"] for pg in pages],
                "token_start": s,
                "token_end": e,
                "token_count": e - s,
            }
            for i, (text, s, e) in enumerate(raw_chunks)
        ]


def run_chunking(config_path="config.yaml"):
    """Chunk all extracted texts in the corpus."""
    cfg = load_config(config_path)
    text_dir = cfg["paths"]["text_dir"]
    chunk_dir = cfg["paths"]["chunk_dir"]
    chunk_cfg = cfg.get("chunk", {})
    chunk_size = chunk_cfg.get("token_size", 400)
    overlap = chunk_cfg.get("overlap", 50)
    tokenizer_name = chunk_cfg.get("tokenizer", "cl100k_base")
    respect = chunk_cfg.get("respect_sections", True)
    os.makedirs(chunk_dir, exist_ok=True)

    enc = tiktoken.get_encoding(tokenizer_name)

    text_files = sorted(f for f in os.listdir(text_dir) if f.endswith(".json"))
    print(f"[chunk] Processing {len(text_files)} papers (size={chunk_size}, overlap={overlap})")
    total_chunks = 0
    success, skipped_count = 0, 0

    for i, fname in enumerate(text_files):
        paper_id = fname.replace(".json", "")
        if already_processed(paper_id, chunk_dir):
            skipped_count += 1
            continue

        text_data = load_json(os.path.join(text_dir, fname))
        chunks = chunk_paper(text_data, enc, chunk_size, overlap, respect)

        output = {
            "paper_id": paper_id,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "num_chunks": len(chunks),
            "chunks": chunks,
        }
        save_json(output, os.path.join(chunk_dir, f"{paper_id}.json"))
        total_chunks += len(chunks)
        success += 1
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(text_files)}] {total_chunks} chunks so far")

    print(f"\nDone: {success} papers, {total_chunks} total chunks, {skipped_count} skipped")


def main():
    ap = argparse.ArgumentParser(description="Chunk extracted texts")
    ap.add_argument("-c", "--config", default="config.yaml")
    args = ap.parse_args()
    run_chunking(args.config)


if __name__ == "__main__":
    main()
