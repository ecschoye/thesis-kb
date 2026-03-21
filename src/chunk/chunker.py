"""Token-based chunking with section awareness and sentence-boundary flex."""
import os, re, argparse
import tiktoken
from src.utils import load_config, load_json, save_json, already_processed

# Patterns for detecting protected blocks that should not be split
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|.*\|\s*$")
_LATEX_BLOCK_START = re.compile(r"\\begin\{(equation|align|table|figure|lstlisting|verbatim|itemize|enumerate)\*?\}")
_LATEX_BLOCK_END = re.compile(r"\\end\{(equation|align|table|figure|lstlisting|verbatim|itemize|enumerate)\*?\}")
_DISPLAY_MATH_RE = re.compile(r"\$\$")

# Sentence-ending punctuation followed by whitespace or newline
_SENTENCE_END_RE = re.compile(r"[.!?]\s")


def _find_protected_spans(text):
    """Find spans of text that should not be split (tables, equations, etc).

    Returns list of (start_char, end_char) tuples.
    """
    spans = []
    lines = text.split("\n")
    char_pos = 0

    # Detect markdown/text table blocks (consecutive rows matching |...|...|)
    table_start = None
    for i, line in enumerate(lines):
        line_start = char_pos
        line_end = char_pos + len(line)
        if _TABLE_ROW_RE.match(line):
            if table_start is None:
                table_start = line_start
        else:
            if table_start is not None:
                spans.append((table_start, line_start))
                table_start = None
        char_pos = line_end + 1  # +1 for newline
    if table_start is not None:
        spans.append((table_start, char_pos))

    # Detect LaTeX block environments
    for m in _LATEX_BLOCK_START.finditer(text):
        env_name = m.group(1)
        end_pattern = re.compile(r"\\end\{" + re.escape(env_name) + r"\*?\}")
        end_match = end_pattern.search(text, m.end())
        if end_match:
            spans.append((m.start(), end_match.end()))

    # Detect $$...$$ display math
    math_starts = list(_DISPLAY_MATH_RE.finditer(text))
    for i in range(0, len(math_starts) - 1, 2):
        spans.append((math_starts[i].start(), math_starts[i + 1].end()))

    # Merge overlapping spans
    if not spans:
        return []
    spans.sort()
    merged = [spans[0]]
    for s, e in spans[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _char_to_token_idx(text, char_pos, enc):
    """Map a character position to the approximate token index."""
    prefix = text[:char_pos]
    return len(enc.encode(prefix))


def token_chunks(text, enc, chunk_size=400, overlap=50, flex_pct=0.15,
                 protect_blocks=True):
    """Split text into overlapping token windows with sentence-boundary flex.

    flex_pct: allow chunk boundaries to flex ±this fraction to hit sentence ends.
    protect_blocks: keep tables/equations/display-math intact within chunks.
    Returns list of (text, start_tok, end_tok) tuples.
    """
    tokens = enc.encode(text)
    total_tokens = len(tokens)

    if total_tokens <= chunk_size:
        return [(text, 0, total_tokens)]

    # Pre-compute protected character spans and map to token ranges
    protected_tok_spans = []
    if protect_blocks:
        char_spans = _find_protected_spans(text)
        for cs, ce in char_spans:
            ts = _char_to_token_idx(text, cs, enc)
            te = _char_to_token_idx(text, ce, enc)
            protected_tok_spans.append((ts, te))

    flex_size = int(chunk_size * flex_pct)
    chunks = []
    start = 0

    while start < total_tokens:
        # Tentative end
        end = min(start + chunk_size, total_tokens)

        if end < total_tokens:
            # Try to find a sentence boundary in the flex zone
            # Search backward from end to (end - flex_size)
            search_start = max(start, end - flex_size)
            chunk_text_candidate = enc.decode(tokens[search_start:end])

            # Find the last sentence-ending position in this zone
            best_offset = None
            for m in _SENTENCE_END_RE.finditer(chunk_text_candidate):
                # Position within the search zone (character-level)
                best_offset = m.start() + 1  # include the punctuation

            if best_offset is not None:
                # Map character offset back to token position
                prefix_text = chunk_text_candidate[:best_offset]
                tok_adjustment = len(enc.encode(prefix_text))
                new_end = search_start + tok_adjustment
                # Only use if we're not shrinking too much
                if new_end > start + int(chunk_size * (1 - flex_pct)):
                    end = new_end

            # Check if end falls inside a protected block — extend to block end
            for ps, pe in protected_tok_spans:
                if ps < end <= pe:
                    # Don't extend beyond 2× chunk_size
                    if pe <= start + chunk_size * 2:
                        end = pe
                    break

        chunk_text = enc.decode(tokens[start:end])
        chunks.append((chunk_text, start, end))

        if end >= total_tokens:
            break
        start = end - overlap

    return chunks


def chunk_paper(text_data, enc, chunk_size=400, overlap=50, respect_sections=True,
                flex_pct=0.15, protect_blocks=True):
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
            raw_chunks = token_chunks(
                sec["text"], enc, chunk_size, overlap,
                flex_pct=flex_pct, protect_blocks=protect_blocks,
            )
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
        raw_chunks = token_chunks(
            full_text, enc, chunk_size, overlap,
            flex_pct=flex_pct, protect_blocks=protect_blocks,
        )
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
    flex_pct = chunk_cfg.get("flex_pct", 0.15)
    protect_blocks = chunk_cfg.get("protect_blocks", True)
    os.makedirs(chunk_dir, exist_ok=True)

    enc = tiktoken.get_encoding(tokenizer_name)

    text_files = sorted(f for f in os.listdir(text_dir) if f.endswith(".json"))
    print(f"[chunk] Processing {len(text_files)} papers (size={chunk_size}, overlap={overlap}, flex={flex_pct})")
    total_chunks = 0
    success, skipped_count = 0, 0

    for i, fname in enumerate(text_files):
        paper_id = fname.replace(".json", "")
        if already_processed(paper_id, chunk_dir):
            skipped_count += 1
            continue

        text_data = load_json(os.path.join(text_dir, fname))
        chunks = chunk_paper(
            text_data, enc, chunk_size, overlap, respect,
            flex_pct=flex_pct, protect_blocks=protect_blocks,
        )

        output = {
            "paper_id": paper_id,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "flex_pct": flex_pct,
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
