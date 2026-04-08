#!/usr/bin/env python3
"""Thesis-grounded evaluation: parse LaTeX paragraphs with \\cite{} keys,
use claim text as queries, check whether cited papers appear in top-k.

Usage:
    python eval/thesis_grounded_eval.py                        # default chapters
    python eval/thesis_grounded_eval.py --chapters 4 5         # specific chapters
    python eval/thesis_grounded_eval.py --k 10 --verbose       # top-10, show details
    python eval/thesis_grounded_eval.py -o eval/thesis_eval.json  # save results
"""
import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

THESIS_DIR = os.path.expanduser("~/TDT4900-master-thesis")
CHAPTERS_DIR = os.path.join(THESIS_DIR, "chapters")
BIB_PATH = os.path.join(THESIS_DIR, "bibtex", "bibliography.bib")

# Chapters with substantive citations (skip abstract, preface)
DEFAULT_CHAPTERS = ["4-background", "5-sota", "6-architecture", "7-experiments", "8-evaluation"]


def parse_bib_keys(bib_path):
    """Parse bib file to get all defined cite keys."""
    keys = set()
    with open(bib_path) as f:
        for line in f:
            m = re.match(r'@\w+\{([^,]+),', line)
            if m:
                keys.add(m.group(1).strip())
    return keys


def extract_citation_sentences(tex_path):
    """Extract sentences containing \\cite{} from a LaTeX file.

    Returns list of dicts with:
    - text: the sentence/clause with citations removed (the "query")
    - cite_keys: list of cite keys referenced
    - line: source line number
    - raw: original text with citations
    """
    with open(tex_path) as f:
        content = f.read()

    # Remove LaTeX comments
    content = re.sub(r'%.*$', '', content, flags=re.MULTILINE)

    # Remove common LaTeX commands that aren't content
    content = re.sub(r'\\(label|ref|eqref|gls|glspl|acrshort|acrlong)\{[^}]*\}', '', content)
    content = re.sub(r'\\(begin|end)\{[^}]*\}', '', content)
    content = re.sub(r'\\(section|subsection|subsubsection|paragraph|chapter)\*?\{[^}]*\}', '', content)
    content = re.sub(r'\\(noindent|medskip|bigskip|newpage|clearpage)', '', content)

    # Find all sentences/clauses containing \cite{}
    # Split on sentence boundaries (period followed by space/newline and capital letter)
    # but keep \cite{} references
    cite_pattern = re.compile(r'\\cite\{([^}]+)\}')

    results = []
    # Split into paragraphs first (double newline)
    paragraphs = re.split(r'\n\s*\n', content)

    for para in paragraphs:
        para = para.strip()
        if not para or len(para) < 30:
            continue

        # Find all citations in this paragraph
        cites_in_para = cite_pattern.findall(para)
        if not cites_in_para:
            continue

        # Split paragraph into sentences (approximate)
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', para)

        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 20:
                continue

            # Find citations in this sentence
            sent_cites = cite_pattern.findall(sent)
            if not sent_cites:
                continue

            # Extract all cite keys (may be comma-separated)
            cite_keys = []
            for cite_group in sent_cites:
                for key in cite_group.split(','):
                    key = key.strip()
                    if key:
                        cite_keys.append(key)

            # Remove \cite{} from text to get the claim
            claim_text = cite_pattern.sub('', sent)
            # Clean up LaTeX artifacts
            claim_text = re.sub(r'~', ' ', claim_text)
            claim_text = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', claim_text)  # \textit{x} -> x
            claim_text = re.sub(r'[$].*?[$]', '', claim_text)  # remove inline math
            claim_text = re.sub(r'\s+', ' ', claim_text).strip()
            claim_text = claim_text.strip('., ')

            if len(claim_text) < 20:
                continue

            results.append({
                "text": claim_text,
                "cite_keys": cite_keys,
                "raw": sent[:200],
            })

    return results


def resolve_cite_keys_to_paper_ids(cite_keys, kb):
    """Map bib cite keys to KB paper_ids using the same logic as api.py."""
    resolved = {}

    for key in cite_keys:
        # Try direct SQLite lookup by matching against title/authors from bib
        # Use normalized matching: lowercase, strip punctuation
        norm_key = re.sub(r'[^a-z0-9]', '', key.lower())

        # Strategy 1: Check if any paper_id contains parts of the cite key
        # Cite keys like "Lee_2020:spike-flownet" map to paper_ids like "Lee_et_al._-_2020_-_Spike-FlowNet_..."
        parts = re.split(r'[_:\-]', key)
        author = parts[0] if parts else ""
        year = None
        for p in parts:
            if re.match(r'^\d{4}$', p):
                year = int(p)
                break

        if author and year:
            rows = kb.db.execute(
                "SELECT DISTINCT paper_id FROM papers WHERE paper_id LIKE ? AND year = ?",
                (f"{author}%", year),
            ).fetchall()
            if len(rows) == 1:
                resolved[key] = rows[0][0]
                continue
            elif len(rows) > 1:
                # Multiple matches — try to disambiguate by other key parts
                key_words = set(p.lower() for p in parts if len(p) > 2 and not p.isdigit())
                best = None
                best_score = 0
                for r in rows:
                    pid = r[0].lower()
                    score = sum(1 for w in key_words if w in pid)
                    if score > best_score:
                        best_score = score
                        best = r[0]
                if best and best_score > 0:
                    resolved[key] = best
                    continue

        # Strategy 2: Search by title words from cite key
        title_words = [p for p in parts if len(p) > 3 and not p.isdigit()]
        if title_words:
            like_pattern = "%".join(title_words[:3])
            rows = kb.db.execute(
                "SELECT paper_id FROM papers WHERE LOWER(title) LIKE ?",
                (f"%{like_pattern.lower()}%",),
            ).fetchall()
            if len(rows) == 1:
                resolved[key] = rows[0][0]

    return resolved


def run_thesis_eval(claims, kb, embed_client, embed_model, instruction, k=20,
                    verbose=False, progress_interval=20):
    """Run retrieval for each claim and check if cited papers appear."""
    from src.embed.embedder import embed_batch

    results = []
    t0 = time.time()

    for i, claim in enumerate(claims):
        query = claim["text"]
        expected_pids = set(claim.get("resolved_paper_ids", {}).values())

        if not expected_pids:
            results.append({
                "query": query[:100],
                "cite_keys": claim["cite_keys"],
                "expected_papers": 0,
                "found_papers": 0,
                "recall": None,
                "skipped": True,
            })
            continue

        # Embed and query
        query_text = f"Instruct: {instruction}\nQuery: {query}" if instruction else query
        try:
            emb = embed_batch(embed_client, [query_text], embed_model)
            search_results = kb.collection.query(
                query_embeddings=emb,
                n_results=k,
                include=["metadatas"],
            )
        except Exception as e:
            results.append({
                "query": query[:100],
                "cite_keys": claim["cite_keys"],
                "expected_papers": len(expected_pids),
                "found_papers": 0,
                "recall": 0.0,
                "error": str(e),
            })
            continue

        # Extract retrieved paper_ids
        retrieved_pids = set()
        for meta in search_results["metadatas"][0]:
            pid = meta.get("paper_id", "")
            if pid:
                retrieved_pids.add(pid)

        found = expected_pids & retrieved_pids
        recall = len(found) / len(expected_pids) if expected_pids else 0.0

        results.append({
            "query": query[:100],
            "cite_keys": claim["cite_keys"],
            "expected_papers": len(expected_pids),
            "expected_pids": list(expected_pids),
            "found_papers": len(found),
            "found_pids": list(found),
            "missing_pids": list(expected_pids - retrieved_pids),
            "recall": recall,
        })

        if progress_interval and (i + 1) % progress_interval == 0:
            elapsed = time.time() - t0
            tested = [r for r in results if not r.get("skipped")]
            avg_recall = sum(r["recall"] for r in tested) / len(tested) if tested else 0
            sys.stderr.write(
                f"\r\033[K  [{i+1}/{len(claims)}] paper_recall={avg_recall:.3f}"
            )
            sys.stderr.flush()

    if progress_interval:
        sys.stderr.write("\r\033[K")
        sys.stderr.flush()

    return results


def main():
    ap = argparse.ArgumentParser(description="Thesis-grounded retrieval evaluation")
    ap.add_argument("--chapters", nargs="*", default=None,
                    help="Chapter numbers or names (default: 4,5,6,7,8)")
    ap.add_argument("--k", type=int, default=20, help="Top-k to check")
    ap.add_argument("--config", "-c", default=None, help="Config file")
    ap.add_argument("--verbose", action="store_true", help="Show per-claim results")
    ap.add_argument("--output", "-o", default=None, help="Save results as JSON")
    ap.add_argument("--list-keys", action="store_true", help="List cite keys and exit")
    args = ap.parse_args()

    if not os.path.isdir(THESIS_DIR):
        print(f"Thesis directory not found: {THESIS_DIR}")
        sys.exit(1)

    # Determine chapters
    if args.chapters:
        chapters = []
        for c in args.chapters:
            if c.isdigit():
                # Find matching chapter file
                matches = list(Path(CHAPTERS_DIR).glob(f"{c}-*.tex"))
                chapters.extend(str(m) for m in matches)
            else:
                chapters.append(os.path.join(CHAPTERS_DIR, f"{c}.tex"))
    else:
        chapters = [os.path.join(CHAPTERS_DIR, f"{c}.tex") for c in DEFAULT_CHAPTERS]

    chapters = [c for c in chapters if os.path.exists(c)]
    if not chapters:
        print("No chapter files found")
        sys.exit(1)

    print(f"Chapters: {[os.path.basename(c) for c in chapters]}")

    # Extract citation sentences
    all_claims = []
    for ch_path in chapters:
        ch_name = os.path.basename(ch_path).replace('.tex', '')
        claims = extract_citation_sentences(ch_path)
        for c in claims:
            c["chapter"] = ch_name
        all_claims.extend(claims)
        print(f"  {ch_name}: {len(claims)} citation sentences")

    print(f"Total citation sentences: {len(all_claims)}")

    # Resolve cite keys to paper_ids
    config_path = args.config or os.environ.get("THESIS_KB_CONFIG", "config-ollama.yaml")
    from src.utils import load_config
    cfg = load_config(config_path)
    from src.query import ThesisKB
    kb = ThesisKB(config_path)

    all_cite_keys = set()
    for c in all_claims:
        all_cite_keys.update(c["cite_keys"])

    print(f"\nResolving {len(all_cite_keys)} unique cite keys to paper_ids...")
    key_to_pid = resolve_cite_keys_to_paper_ids(all_cite_keys, kb)
    print(f"  Resolved: {len(key_to_pid)} / {len(all_cite_keys)}")

    if args.list_keys:
        print("\nResolved keys:")
        for k, v in sorted(key_to_pid.items()):
            print(f"  {k:50s} -> {v[:50]}")
        unresolved = all_cite_keys - set(key_to_pid.keys())
        if unresolved:
            print(f"\nUnresolved ({len(unresolved)}):")
            for k in sorted(unresolved):
                print(f"  {k}")
        return

    # Attach resolved paper_ids to claims
    for claim in all_claims:
        claim["resolved_paper_ids"] = {
            k: key_to_pid[k] for k in claim["cite_keys"] if k in key_to_pid
        }

    testable = [c for c in all_claims if c["resolved_paper_ids"]]
    print(f"Testable claims (with resolved paper_ids): {len(testable)} / {len(all_claims)}")

    if not testable:
        print("No testable claims found. Run with --list-keys to debug resolution.")
        sys.exit(1)

    # Run evaluation
    from src.embed.embedder import make_embed_client
    embed_client, embed_model = make_embed_client(cfg)
    instruction = cfg.get("embed", {}).get("embedding", {}).get("query_instruction", "")

    print(f"\nRunning retrieval eval (k={args.k})...")
    t0 = time.time()
    results = run_thesis_eval(testable, kb, embed_client, embed_model, instruction,
                              k=args.k, verbose=args.verbose)
    elapsed = time.time() - t0

    # Compute metrics
    tested = [r for r in results if not r.get("skipped") and r.get("recall") is not None]
    if not tested:
        print("No results to evaluate")
        sys.exit(1)

    avg_recall = sum(r["recall"] for r in tested) / len(tested)
    perfect_recall = sum(1 for r in tested if r["recall"] == 1.0) / len(tested)
    zero_recall = sum(1 for r in tested if r["recall"] == 0.0) / len(tested)

    # Per-chapter breakdown
    chapter_metrics = defaultdict(lambda: {"tested": 0, "total_recall": 0.0, "perfect": 0, "zero": 0})
    for claim, result in zip(testable, results):
        if result.get("skipped") or result.get("recall") is None:
            continue
        ch = claim.get("chapter", "unknown")
        chapter_metrics[ch]["tested"] += 1
        chapter_metrics[ch]["total_recall"] += result["recall"]
        if result["recall"] == 1.0:
            chapter_metrics[ch]["perfect"] += 1
        if result["recall"] == 0.0:
            chapter_metrics[ch]["zero"] += 1

    # Display
    print(f"\n{'=' * 60}")
    print(f"Thesis-Grounded Eval (n={len(tested)}, k={args.k}, {elapsed:.1f}s)")
    print(f"{'=' * 60}")
    print(f"  Mean paper recall:      {avg_recall:.4f}")
    print(f"  Perfect recall (100%):  {perfect_recall:.4f} ({sum(1 for r in tested if r['recall']==1.0)}/{len(tested)})")
    print(f"  Zero recall (0%):       {zero_recall:.4f} ({sum(1 for r in tested if r['recall']==0.0)}/{len(tested)})")
    print()

    print("By chapter:")
    print(f"  {'Chapter':25s} {'Tested':>6s} {'Recall':>8s} {'Perfect':>8s} {'Zero':>8s}")
    print(f"  {'-' * 57}")
    for ch, m in sorted(chapter_metrics.items()):
        avg = m["total_recall"] / m["tested"] if m["tested"] else 0
        print(f"  {ch:25s} {m['tested']:6d} {avg:8.4f} {m['perfect']:8d} {m['zero']:8d}")

    if args.verbose:
        print(f"\nFailed retrievals (recall=0):")
        for r in tested:
            if r["recall"] == 0.0:
                print(f"  Q: {r['query'][:80]}")
                print(f"     Keys: {r['cite_keys']}")
                print(f"     Missing: {r.get('missing_pids', [])}")

    # Save output
    if args.output:
        output = {
            "config": {"k": args.k, "chapters": [os.path.basename(c) for c in chapters]},
            "metrics": {
                "n_tested": len(tested),
                "mean_recall": round(avg_recall, 4),
                "perfect_recall_pct": round(perfect_recall, 4),
                "zero_recall_pct": round(zero_recall, 4),
            },
            "chapter_metrics": {ch: {**m, "avg_recall": round(m["total_recall"]/m["tested"], 4) if m["tested"] else 0}
                                for ch, m in chapter_metrics.items()},
            "elapsed_s": round(elapsed, 2),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "per_claim": results,
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output}")

    sys.exit(0 if avg_recall > 0.5 else 1)


if __name__ == "__main__":
    main()
