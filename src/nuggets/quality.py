"""LLM-based nugget quality checking via vLLM OpenAI-compatible API."""
import os, json, time, argparse, threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from src.utils import load_config, load_json, save_json


QUALITY_SYSTEM_PROMPT = """You are a research nugget quality auditor. You will receive a batch of question-answer nuggets extracted from an academic paper. Rate each nugget on 6 dimensions (1=poor, 5=excellent):

1. relevance: Is the Q&A about substantive research content (methods, results, claims)? Score 1 if it covers trivial metadata, paper structure, or generic boilerplate.
2. specificity: Does the answer contain specific details — numbers, method names, dataset names, concrete claims? Score 1 if the answer is vague or generic.
3. self_contained: Can a reader understand this nugget without reading the source paper? Score 1 if it relies on undefined references like "the proposed method" or "Table 3" without context.
4. type_accuracy: Does the assigned type label (method/result/claim/limitation/comparison/background) correctly describe this nugget? Score 1 if clearly wrong.
5. coherence: Is the question well-formed and does the answer directly address it? Score 1 if the Q&A is malformed, circular, or contradictory.
6. thesis_relevance: How relevant is this nugget to a thesis on RGB-Event camera fusion for object detection on resource-constrained platforms? Score 5 for core topics (SNNs, event cameras, RGB-Event fusion, motion compensation, optical flow, neuromorphic hardware, low-latency object detection, trajectory prediction, spike encoding, energy-efficient inference). Score 3 for related but peripheral topics (general object detection, CNN architectures used as baselines, standard datasets). Score 1 for unrelated content (NLP methods, medical imaging, etc.).

Also provide:
- overall: the minimum of dimensions 1-5 (NOT including thesis_relevance)
- flags: array of short string labels for issues found (e.g. "too_vague", "wrong_type", "not_self_contained", "trivial_metadata", "malformed_qa", "missing_specifics", "off_topic"). Use empty array [] if no issues.

Output ONLY a JSON array with one object per nugget, in the same order as the input."""


QUALITY_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "idx": {"type": "integer"},
            "relevance": {"type": "integer", "minimum": 1, "maximum": 5},
            "specificity": {"type": "integer", "minimum": 1, "maximum": 5},
            "self_contained": {"type": "integer", "minimum": 1, "maximum": 5},
            "type_accuracy": {"type": "integer", "minimum": 1, "maximum": 5},
            "coherence": {"type": "integer", "minimum": 1, "maximum": 5},
            "thesis_relevance": {"type": "integer", "minimum": 1, "maximum": 5},
            "overall": {"type": "integer", "minimum": 1, "maximum": 5},
            "flags": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "idx", "relevance", "specificity", "self_contained",
            "type_accuracy", "coherence", "thesis_relevance", "overall", "flags",
        ],
    },
}


def build_quality_prompt(batch, paper_title, paper_id):
    """Format a batch of nuggets into a numbered list for quality rating."""
    lines = [f'Rate these nuggets from paper "{paper_title}" ({paper_id}):\n']
    for i, n in enumerate(batch):
        lines.append(
            f"[{i + 1}] type={n.get('type', 'unknown')}\n"
            f"Q: {n['question']}\n"
            f"A: {n['answer']}\n"
        )
    return "\n".join(lines)


def rate_nugget_batch(client, batch, model, paper_title, paper_id, cfg):
    """Send a batch of nuggets to the LLM for quality rating."""
    prompt = build_quality_prompt(batch, paper_title, paper_id)
    max_tokens = cfg.get("max_tokens", 800)
    temperature = cfg.get("temperature", 0.0)
    max_retries = cfg.get("max_retries", 3)
    retry_delay = cfg.get("retry_base_delay", 2.0)

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": QUALITY_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "nugget_ratings",
                        "schema": QUALITY_SCHEMA,
                        "strict": False,
                    },
                },
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            raw = resp.choices[0].message.content
            scores = json.loads(raw)

            # Validate we got the right number of results
            if not isinstance(scores, list):
                return None, f"Expected array, got {type(scores).__name__}"

            # Map scores back to nugget_ids
            results = []
            for i, n in enumerate(batch):
                if i < len(scores):
                    s = scores[i]
                    results.append({
                        "nugget_id": n.get("nugget_id", f"unknown_{i}"),
                        "relevance": s.get("relevance", 0),
                        "specificity": s.get("specificity", 0),
                        "self_contained": s.get("self_contained", 0),
                        "type_accuracy": s.get("type_accuracy", 0),
                        "coherence": s.get("coherence", 0),
                        "thesis_relevance": s.get("thesis_relevance", 0),
                        "overall": s.get("overall", 0),
                        "flags": s.get("flags", []),
                    })
                else:
                    # LLM returned fewer results than expected
                    results.append({
                        "nugget_id": n.get("nugget_id", f"unknown_{i}"),
                        "relevance": 0, "specificity": 0, "self_contained": 0,
                        "type_accuracy": 0, "coherence": 0, "thesis_relevance": 0,
                        "overall": 0, "flags": ["rating_missing"],
                    })
            return results, None

        except Exception as e:
            err_str = str(e)
            print(f"  RETRY {paper_id} batch attempt {attempt+1}/{max_retries}: {err_str[:200]}", flush=True)
            if "rate" in err_str.lower() and attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt))
            elif attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                return None, err_str

    return None, "max retries exceeded"


def _process_paper(client, paper_id, nuggets, model, paper_title, cfg):
    """Rate all nuggets for a single paper in sequential batches."""
    batch_size = cfg.get("batch_size", 10)
    all_scores = []
    warnings = []

    for i in range(0, len(nuggets), batch_size):
        batch = nuggets[i : i + batch_size]
        results, err = rate_nugget_batch(
            client, batch, model, paper_title, paper_id, cfg
        )
        if results:
            all_scores.extend(results)
        else:
            warnings.append(f"  WARN batch {i // batch_size} of {paper_id}: {err}")
            # Fill with zeros for failed batch
            for n in batch:
                all_scores.append({
                    "nugget_id": n.get("nugget_id", "unknown"),
                    "relevance": 0, "specificity": 0, "self_contained": 0,
                    "type_accuracy": 0, "coherence": 0, "thesis_relevance": 0,
                    "overall": 0, "flags": ["batch_failed"],
                })

    return all_scores, warnings


def run_quality_check(config_path="config.yaml"):
    """Run quality checks on all extracted nuggets."""
    cfg = load_config(config_path)
    nugget_dir = cfg["paths"]["nugget_dir"]
    quality_dir = cfg["paths"].get("quality_dir", "corpus/nuggets_quality")
    ncfg = cfg.get("nuggets", {})
    qcfg = ncfg.get("quality", {})
    max_workers = qcfg.get("max_workers", 48)
    flag_threshold = qcfg.get("flag_threshold", 2)
    os.makedirs(quality_dir, exist_ok=True)

    # Set up LLM client (same backend logic as extract.py)
    backend = ncfg.get("backend", "vllm")
    if backend == "ollama":
        ollama_cfg = ncfg.get("ollama", {})
        base_url = ollama_cfg.get("base_url", "http://127.0.0.1:11434/v1")
        model = ollama_cfg.get("model", "qwen3.5:27b")
        client = OpenAI(base_url=base_url, api_key="ollama")
    else:
        vllm_cfg = ncfg.get("vllm", {})
        port = vllm_cfg.get("port", 8000)
        model = vllm_cfg.get("model", "Qwen/Qwen3.5-27B")
        client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="none")

    # Find papers to process
    def _quality_done(paper_id):
        path = os.path.join(quality_dir, f"{paper_id}.json")
        if not os.path.exists(path):
            return False
        try:
            with open(path) as f:
                data = json.load(f)
            return data.get("num_nuggets", 0) > 0
        except (json.JSONDecodeError, OSError):
            return False

    nugget_files = sorted(f for f in os.listdir(nugget_dir) if f.endswith(".json"))
    to_process = []
    skipped_count = 0
    for fname in nugget_files:
        paper_id = fname.replace(".json", "")
        if _quality_done(paper_id):
            skipped_count += 1
        else:
            to_process.append(fname)

    print(f"[quality] Processing {len(to_process)} papers ({skipped_count} skipped) via {model}, workers={max_workers}")

    # Load all paper nuggets
    paper_data = {}
    load_failed = 0
    for fname in to_process:
        paper_id = fname.replace(".json", "")
        try:
            data = load_json(os.path.join(nugget_dir, fname))
            nuggets = data.get("nuggets", [])
            if nuggets:
                paper_data[paper_id] = {
                    "nuggets": nuggets,
                    "paper_title": nuggets[0].get("paper_title", paper_id),
                }
            else:
                skipped_count += 1
        except Exception as e:
            print(f"  ERROR loading nuggets for {paper_id}: {e}")
            load_failed += 1

    total_papers = len(paper_data)
    total_nuggets = 0
    total_flagged = 0
    success = 0
    failed = 0
    papers_done = 0
    lock = threading.Lock()

    def _on_paper_done(fut, paper_id):
        nonlocal total_nuggets, total_flagged, success, failed, papers_done
        try:
            scores, warnings = fut.result()
        except Exception as e:
            with lock:
                failed += 1
                papers_done += 1
                print(f"  ERROR {paper_id}: {e}")
            return

        # Compute summary stats
        valid_scores = [s for s in scores if s["overall"] > 0]
        mean_overall = (
            sum(s["overall"] for s in valid_scores) / len(valid_scores)
            if valid_scores
            else 0.0
        )
        num_flagged = sum(1 for s in scores if s["overall"] <= flag_threshold)

        output = {
            "paper_id": paper_id,
            "num_nuggets": len(scores),
            "mean_overall": round(mean_overall, 2),
            "num_flagged": num_flagged,
            "scores": scores,
        }
        save_json(output, os.path.join(quality_dir, f"{paper_id}.json"))

        with lock:
            total_nuggets += len(scores)
            total_flagged += num_flagged
            success += 1
            papers_done += 1
            for w in warnings:
                print(w)
            print(
                f"  [{papers_done}/{total_papers}] {paper_id[:40]:40s} "
                f"-> {len(scores)} nuggets, mean={mean_overall:.1f}, flagged={num_flagged}"
            )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for paper_id, pdata in paper_data.items():
            fut = executor.submit(
                _process_paper,
                client, paper_id, pdata["nuggets"], model, pdata["paper_title"], qcfg,
            )
            fut.add_done_callback(
                lambda f, pid=paper_id: _on_paper_done(f, pid)
            )

    print(
        f"\nDone: {success} papers, {total_nuggets} nuggets rated, "
        f"{total_flagged} flagged (overall<={flag_threshold}), "
        f"{failed} failed, {skipped_count} skipped"
    )


def main():
    ap = argparse.ArgumentParser(description="Quality-check extracted nuggets")
    ap.add_argument("-c", "--config", default="config.yaml")
    args = ap.parse_args()
    run_quality_check(args.config)


if __name__ == "__main__":
    main()
