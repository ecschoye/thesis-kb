"""LLM-based nugget augmentation: improve weak nuggets and fill coverage gaps."""
import os, json, re, time, argparse, threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from src.utils import load_config, load_json, save_json, make_llm_client
from src.nuggets.extract import repair_json


IMPROVE_SYSTEM_PROMPT = """You are a research knowledge refinement specialist. You will receive:
1. A weak nugget (question-answer pair) extracted from an academic paper
2. The quality issues identified with this nugget
3. The original source text from which this nugget was extracted

Your task: produce an IMPROVED version of this nugget that fixes the identified issues.

Rules:
- Fix the specific issues listed (e.g., add missing numbers, make self-contained, add method names)
- Keep the same general topic — do not change what the nugget is about
- The improved nugget must be self-contained (understandable without the source paper)
- Replace vague references like "the proposed method" with the actual method name
- Include specific numbers, dataset names, metric values when available in the source text
- If the source text does not contain enough information to fix the issue, return the original nugget unchanged and set "improved": false
- Preserve the nugget type unless it was flagged as wrong_type
- Output ONLY valid JSON, no markdown fences, no preamble

Output format:
{
  "question": "...",
  "answer": "...",
  "type": "method|result|claim|limitation|comparison|background",
  "improved": true,
  "changes": "brief description of what was changed"
}"""


GAPFILL_SYSTEM_PROMPT = """You are a research knowledge extractor for a thesis on RGB-Event camera fusion for object detection on resource-constrained autonomous platforms.

You will receive:
1. A chunk of academic text
2. Nuggets that have ALREADY been extracted from this text

Your task: identify important factual content in the text that is NOT covered by the existing nuggets.

Rules:
- Only extract NEW information not already covered by existing nuggets
- Focus on SUBSTANTIVE content: methods, results, comparisons, limitations, specific numbers
- Prioritize thesis-relevant gaps: spike encoding details, event representations, fusion mechanisms, SNN parameters, energy/latency metrics, motion compensation techniques, optical flow, neuromorphic hardware results, trajectory prediction
- Each nugget must be self-contained (understandable without the source text)
- Include specific numbers: accuracies, FPS, energy, parameters, latency, FLOPs
- Classify type: method, result, claim, limitation, comparison, background
- If the existing nuggets already cover everything important, return an empty array []
- Do NOT rephrase existing nuggets — only add genuinely new information
- Aim for 1-5 new nuggets. Prefer fewer, higher-quality nuggets.
- Output ONLY valid JSON array (or empty array []), no markdown fences, no preamble

Output format:
[
  {
    "question": "...",
    "answer": "...",
    "type": "method|result|claim|limitation|comparison|background"
  }
]"""


def _is_reference_chunk(text):
    """Heuristic: chunk is mostly bibliographic references."""
    lines = text.strip().split("\n")
    if not lines:
        return True
    ref_lines = sum(1 for l in lines if re.match(r"^\s*\[?\d+\]?\s", l))
    return ref_lines / len(lines) > 0.4


def improve_nugget(client, nugget, scores, chunk_text, model, cfg):
    """Send a weak nugget + quality feedback + source to the LLM for improvement."""
    temperature = cfg.get("temperature", 0.1)
    max_tokens = cfg.get("max_tokens", 2000)
    max_retries = cfg.get("max_retries", 3)
    retry_delay = cfg.get("retry_base_delay", 2.0)

    flags_str = ", ".join(scores.get("flags", [])) or "none"
    user_msg = (
        f"Improve this nugget:\n"
        f"Q: {nugget['question']}\n"
        f"A: {nugget['answer']}\n"
        f"Type: {nugget.get('type', 'unknown')}\n\n"
        f"Quality issues: {flags_str}\n"
        f"Scores: specificity={scores.get('specificity', '?')}, "
        f"self_contained={scores.get('self_contained', '?')}\n\n"
        f"Source text from the paper:\n{chunk_text}"
    )

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": IMPROVE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            raw = resp.choices[0].message.content
            parsed = repair_json(raw)
            if parsed and isinstance(parsed, dict) and "question" in parsed:
                return {
                    "question": str(parsed["question"]),
                    "answer": str(parsed["answer"]),
                    "type": str(parsed.get("type", nugget.get("type", "background"))),
                    "improved": bool(parsed.get("improved", True)),
                    "changes": str(parsed.get("changes", "")),
                }
            # If repair_json returned a list, take first element
            if parsed and isinstance(parsed, list) and len(parsed) > 0:
                p = parsed[0]
                if isinstance(p, dict) and "question" in p:
                    return {
                        "question": str(p["question"]),
                        "answer": str(p["answer"]),
                        "type": str(p.get("type", nugget.get("type", "background"))),
                        "improved": bool(p.get("improved", True)),
                        "changes": str(p.get("changes", "")),
                    }
        except Exception as e:
            if "rate" in str(e).lower() and attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt))
                continue
        break

    # Fallback: return original unchanged
    return {
        "question": nugget["question"],
        "answer": nugget["answer"],
        "type": nugget.get("type", "background"),
        "improved": False,
        "changes": "",
    }


def gapfill_chunk(client, chunk_text, existing_nuggets, model, cfg):
    """Ask the LLM to find missed content from a chunk given existing nuggets."""
    temperature = cfg.get("temperature", 0.1)
    max_tokens = cfg.get("max_tokens", 2000)
    max_retries = cfg.get("max_retries", 3)
    retry_delay = cfg.get("retry_base_delay", 2.0)

    nugget_lines = []
    for i, n in enumerate(existing_nuggets):
        nugget_lines.append(
            f"[{i + 1}] Q: {n['question']}\n    A: {n['answer']}"
        )
    existing_str = "\n".join(nugget_lines) if nugget_lines else "(none)"

    user_msg = (
        f"Source text:\n{chunk_text}\n\n"
        f"Already extracted nuggets from this text:\n{existing_str}\n\n"
        f"Extract any important information that was MISSED by the existing nuggets."
    )

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": GAPFILL_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            raw = resp.choices[0].message.content
            parsed = repair_json(raw)
            if parsed is None:
                return []
            if isinstance(parsed, dict):
                parsed = [parsed]
            if not isinstance(parsed, list):
                return []
            valid = []
            for n in parsed:
                if isinstance(n, dict) and "question" in n and "answer" in n:
                    valid.append({
                        "question": str(n["question"]),
                        "answer": str(n["answer"]),
                        "type": str(n.get("type", "background")),
                    })
            return valid
        except Exception as e:
            if "rate" in str(e).lower() and attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt))
                continue
            break

    return []


def _dedup_against_existing(new_nuggets, existing_nuggets, threshold=0.75):
    """Remove new nuggets that overlap >threshold with existing ones."""
    existing_wordsets = []
    for n in existing_nuggets:
        words = set(n["question"].lower().split()) | set(n["answer"].lower().split())
        existing_wordsets.append(words)

    kept = []
    for n in new_nuggets:
        words = set(n["question"].lower().split()) | set(n["answer"].lower().split())
        is_dup = False
        for ew in existing_wordsets:
            overlap = len(words & ew) / max(len(words | ew), 1)
            if overlap > threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(n)
            existing_wordsets.append(words)  # also dedup against each other
    return kept


def _process_paper(client, paper_id, nuggets, quality_data, chunks_data, model, cfg):
    """Run both augmentation passes for a single paper."""
    improve_threshold = cfg.get("improve_threshold", 2)
    gap_max_nuggets = cfg.get("gap_max_nuggets", 2)
    gap_min_tokens = cfg.get("gap_min_tokens", 100)
    gap_skip_sections = set(cfg.get("gap_skip_sections", ["references", "acknowledgments", "bibliography"]))

    # Build lookups
    quality_by_id = {}
    for s in quality_data.get("scores", []):
        quality_by_id[s["nugget_id"]] = s

    chunk_by_id = {}
    for c in chunks_data.get("chunks", []):
        chunk_by_id[c["chunk_id"]] = c

    nuggets_by_chunk = defaultdict(list)
    for n in nuggets:
        nuggets_by_chunk[n.get("source_chunk")].append(n)

    # --- Pass 1: Improve weak nuggets ---
    improved = []
    for n in nuggets:
        nid = n.get("nugget_id", "")
        scores = quality_by_id.get(nid)
        if not scores or scores.get("overall", 5) > improve_threshold:
            continue
        chunk = chunk_by_id.get(n.get("source_chunk"))
        if not chunk:
            continue

        result = improve_nugget(client, n, scores, chunk["text"], model, cfg)
        if result.get("improved"):
            result["original_nugget_id"] = nid
            result["source_chunk"] = n.get("source_chunk")
            result["section"] = n.get("section", "unknown")
            result["pages"] = n.get("pages", [])
            improved.append(result)

    # --- Pass 2: Gap-fill sparse chunks ---
    all_gap_filled = []
    for chunk_id, chunk in chunk_by_id.items():
        section = chunk.get("section", "")
        if section in gap_skip_sections:
            continue
        if chunk.get("token_count", 0) < gap_min_tokens:
            continue
        if _is_reference_chunk(chunk["text"]):
            continue

        existing = nuggets_by_chunk.get(chunk_id, [])
        if len(existing) > gap_max_nuggets:
            continue

        new_nuggets = gapfill_chunk(client, chunk["text"], existing, model, cfg)
        for gn in new_nuggets:
            gn["source_chunk"] = chunk_id
            gn["section"] = section
            gn["pages"] = chunk.get("pages", [])
        all_gap_filled.extend(new_nuggets)

    # Dedup gap-filled against all existing nuggets
    all_gap_filled = _dedup_against_existing(all_gap_filled, nuggets)

    # Assign IDs to gap-filled nuggets
    for i, gn in enumerate(all_gap_filled):
        gn["nugget_id"] = f"{paper_id}_gap_{i}"

    return {
        "paper_id": paper_id,
        "num_improved": len(improved),
        "num_gap_filled": len(all_gap_filled),
        "improved": improved,
        "gap_filled": all_gap_filled,
    }


def run_augmentation(config_path="config.yaml"):
    """Run augmentation on all papers with nuggets + quality scores."""
    cfg = load_config(config_path)
    nugget_dir = cfg["paths"]["nugget_dir"]
    quality_dir = cfg["paths"].get("quality_dir", "corpus/nuggets_quality")
    chunk_dir = cfg["paths"]["chunk_dir"]
    augmented_dir = cfg["paths"].get("augmented_dir", "corpus/nuggets_augmented")
    ncfg = cfg.get("nuggets", {})
    acfg = ncfg.get("augmentation", {})
    max_workers = acfg.get("max_workers", 32)
    os.makedirs(augmented_dir, exist_ok=True)

    client, model = make_llm_client(cfg)

    # Find papers to process (need nuggets + quality + chunks)
    def _done(paper_id):
        path = os.path.join(augmented_dir, f"{paper_id}.json")
        if not os.path.exists(path):
            return False
        try:
            with open(path) as f:
                data = json.load(f)
            return "improved" in data and "gap_filled" in data
        except (json.JSONDecodeError, OSError):
            return False

    nugget_files = sorted(f for f in os.listdir(nugget_dir) if f.endswith(".json"))
    to_process = []
    skipped = 0
    missing = 0
    for fname in nugget_files:
        paper_id = fname.replace(".json", "")
        if _done(paper_id):
            skipped += 1
            continue
        # Check all three inputs exist
        quality_path = os.path.join(quality_dir, fname)
        chunk_path = os.path.join(chunk_dir, fname)
        if not os.path.exists(quality_path) or not os.path.exists(chunk_path):
            missing += 1
            continue
        to_process.append(paper_id)

    total_papers = len(to_process)
    print(f"[augment] Processing {total_papers} papers ({skipped} skipped, {missing} missing deps) via {model}, workers={max_workers}")

    if not to_process:
        return

    # Process papers concurrently
    total_improved = 0
    total_gap_filled = 0
    success = 0
    failed = 0
    papers_done = 0
    lock = threading.Lock()

    def _on_done(fut, paper_id):
        nonlocal total_improved, total_gap_filled, success, failed, papers_done
        try:
            result = fut.result()
        except Exception as e:
            with lock:
                failed += 1
                papers_done += 1
                print(f"  ERROR {paper_id}: {e}")
            return

        save_json(result, os.path.join(augmented_dir, f"{paper_id}.json"))

        with lock:
            total_improved += result["num_improved"]
            total_gap_filled += result["num_gap_filled"]
            success += 1
            papers_done += 1
            print(
                f"  [{papers_done}/{total_papers}] {paper_id[:40]:40s} "
                f"-> improved={result['num_improved']}, gap_filled={result['num_gap_filled']}"
            )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for paper_id in to_process:
            nuggets_data = load_json(os.path.join(nugget_dir, f"{paper_id}.json"))
            quality_data = load_json(os.path.join(quality_dir, f"{paper_id}.json"))
            chunks_data = load_json(os.path.join(chunk_dir, f"{paper_id}.json"))

            fut = executor.submit(
                _process_paper,
                client, paper_id,
                nuggets_data.get("nuggets", []),
                quality_data,
                chunks_data,
                model, acfg,
            )
            fut.add_done_callback(
                lambda f, pid=paper_id: _on_done(f, pid)
            )

    print(
        f"\nDone: {success} papers, {total_improved} improved, "
        f"{total_gap_filled} gap-filled, {failed} failed, {skipped} skipped"
    )


def main():
    ap = argparse.ArgumentParser(description="Augment existing nuggets")
    ap.add_argument("-c", "--config", default="config.yaml")
    args = ap.parse_args()
    run_augmentation(args.config)


if __name__ == "__main__":
    main()
