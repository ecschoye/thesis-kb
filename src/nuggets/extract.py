"""LLM-based nugget extraction via vLLM OpenAI-compatible API."""
import os, json, re, sys, time, argparse, threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils import load_config, load_json, save_json, make_llm_client


SYSTEM_PROMPT = """You are a research knowledge extractor for a thesis on RGB-Event camera fusion for object detection on resource-constrained autonomous platforms. The thesis covers: spiking neural networks (SNNs), event camera processing, RGB-Event fusion, optical flow estimation, motion compensation, object detection, short-term trajectory prediction, neuromorphic/energy-efficient hardware, and low-latency inference.

Given a chunk of an academic paper, extract the important factual claims as atomic question-answer nuggets.

Rules:
- Extract ALL substantive contributions of the paper (methods, results, comparisons, limitations, key claims)
- Give EXTRA DEPTH to thesis-relevant topics: extract fine-grained details about spike encoding, membrane potentials, event representations (voxel grids, time surfaces, spike tensors), motion compensation, temporal processing, fusion architectures, energy/latency/FPS metrics on embedded hardware, neuromorphic chip results, DVS/DAVIS sensor specifics, optical flow, SNN architectures, trajectory prediction
- For non-thesis topics in the paper, still extract key contributions but at a higher level (fewer nuggets, less granular)
- Skip trivial metadata: affiliations, section headings, acknowledgments
- Skip generic background that any textbook covers — only extract background claims specific to this work
- NEVER repeat information. If a fact was likely covered in a previous chunk (e.g. the paper's main contribution, what the method does at a high level), do NOT extract it again. Each nugget must add NEW information not already captured. When in doubt, skip it.
- Each nugget must be self-contained (understandable without the source text)
- COMBINE related facts into single nuggets rather than splitting them (e.g. one nugget for "FPGA resource usage" covering FFs, LUTs, and BRAM together)
- Include specific numbers: accuracies, FPS, energy, parameters, latency, FLOPs
- Answers should be DETAILED with exact values when available
- Classify type: method, result, claim, limitation, comparison, background
- Skip content that is purely transitional or structural ("The paper is organized as follows...")
- For RESULTS and COMPARISONS: always include the dataset/benchmark name, the specific baseline methods compared against, and the metric used (e.g. "mAP@50 on DSEC" not just "mAP")
- Extract MOTIVATION nuggets: why the authors propose their approach — what gap or limitation of prior work they address. Classify these as "claim"
- For METHOD nuggets: include training details when present (optimizer, learning rate schedule, batch size, epochs, GPU type, training time) — these are needed for reproducibility discussion
- Extract ABLATION results: when authors remove or vary a component and report the effect, capture both what was changed and the resulting delta (e.g. "Removing temporal attention drops mAP by 3.2% on Gen1"). Classify these as "result"
- When the paper title or author names appear in the chunk, weave them into the answer naturally (e.g. "Wang et al. propose..." or "In EV-FlowNet, ..."). This makes nuggets directly citable without metadata lookup
- If the chunk contains the paper title, author list, year, or venue: emit ONE "background" nugget with Q: "What is the title and who are the authors of this paper?" and A containing the full title, all author names, year, and venue/conference. Only emit this ONCE per paper — if the chunk looks like a later section (methods, experiments, conclusion), skip this nugget.
- Do NOT extract nuggets that admit missing information (e.g. "specific details are not provided in this text"). If the chunk doesn't have the details, skip the nugget entirely.
- Do NOT extract nuggets about cited/referenced works from the bibliography section. Only extract knowledge about the paper itself — not what other papers it cites.
- Aim for 3-8 nuggets per chunk. Fewer, higher-quality nuggets are always better than exhaustive coverage. A chunk with only 2-3 strong nuggets is fine.
- Output ONLY valid JSON array, no markdown fences, no preamble

Output format:
[
  {
    "question": "What neuron model is used in the SNN and what are its parameters?",
    "answer": "Leaky integrate-and-fire (LIF) neuron with learnable membrane time constant tau_m initialized to 2.0, threshold voltage V_th=1.0, and soft reset mechanism.",
    "type": "method"
  },
  {
    "question": "How does the proposed method compare to prior SNN methods on PKU-DVS-Gesture?",
    "answer": "The method achieves 97.2% mAP on PKU-DVS-Gesture, outperforming the previous best SNN-based method (94.8%) by 2.4 percentage points.",
    "type": "result"
  }
]"""


def repair_json(text):
    """Attempt to extract valid JSON from potentially malformed LLM output."""
    text = text.strip()
    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to find array in text
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


def extract_nuggets_from_chunk(client, chunk_text, model, temperature=0.1, max_tokens=3000, extra_body=None, prior_questions=None):
    """Send a chunk to the LLM and parse nuggets."""
    try:
        user_content = "Extract the key knowledge nuggets from this academic text:\n\n" + chunk_text
        if prior_questions:
            already = "\n".join(f"- {q}" for q in prior_questions[-20:])  # last 20 to stay in context
            user_content += f"\n\nNuggets ALREADY EXTRACTED from earlier chunks of this paper (do NOT repeat these):\n{already}"
        kwargs = dict(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if extra_body:
            kwargs["extra_body"] = extra_body
        resp = client.chat.completions.create(**kwargs)
        raw = resp.choices[0].message.content
        nuggets = repair_json(raw)
        if nuggets is None:
            return [], raw
        # Validate structure
        valid = []
        for n in nuggets:
            if isinstance(n, dict) and "question" in n and "answer" in n:
                valid.append({
                    "question": str(n["question"]),
                    "answer": str(n["answer"]),
                    "type": str(n.get("type", "background")),
                })
        return valid, raw
    except Exception as e:
        return [], str(e)


def _process_chunk(client, chunk, model, temp, max_tok, max_retries, retry_delay, paper_id, extra_body=None, prior_questions=None):
    """Process a single chunk — designed for use in a thread pool."""
    text = chunk["text"]
    if len(text.strip()) < 50:
        return []

    cur_prior = prior_questions
    for attempt in range(max_retries):
        nuggets, raw = extract_nuggets_from_chunk(client, text, model, temp, max_tok, extra_body=extra_body, prior_questions=cur_prior)
        if nuggets:
            break
        raw_str = str(raw).lower()
        if "rate" in raw_str:
            time.sleep(retry_delay * (2 ** attempt))
        elif "too long" in raw_str or "maximum context" in raw_str or "400" in raw_str or "max_model_len" in raw_str or "prompt is too long" in raw_str:
            # Token overflow — drop prior_questions and retry
            if cur_prior:
                cur_prior = cur_prior[-10:] if len(cur_prior) > 10 else None
            else:
                break
        else:
            break

    for n in nuggets:
        n["paper_id"] = paper_id
        n["source_chunk"] = chunk["chunk_id"]
        n["section"] = chunk.get("section", "unknown")
        n["pages"] = chunk.get("pages", [])
    return nuggets


def _deduplicate_nuggets(all_nuggets, paper_id):
    """Deduplicate nuggets from overlapping chunks (>60% word overlap)."""
    deduped = []
    deduped_wordsets = []
    for n in all_nuggets:
        words = set(n["question"].lower().split()) | set(n["answer"].lower().split())
        is_dup = False
        for kept_words in deduped_wordsets:
            overlap = len(words & kept_words) / max(len(words | kept_words), 1)
            if overlap > 0.60:
                is_dup = True
                break
        if not is_dup:
            deduped.append(n)
            deduped_wordsets.append(words)
    for i, n in enumerate(deduped):
        n["nugget_id"] = f"{paper_id}_{i}"
    return deduped


def run_extraction(config_path="config.yaml"):
    """Extract nuggets from all chunked papers."""
    cfg = load_config(config_path)
    chunk_dir = cfg["paths"]["chunk_dir"]
    nugget_dir = cfg["paths"]["nugget_dir"]
    ncfg = cfg.get("nuggets", {})
    ext_cfg = ncfg.get("extraction", {})
    max_workers = ext_cfg.get("max_workers", 8)
    os.makedirs(nugget_dir, exist_ok=True)

    client, model = make_llm_client(cfg)

    # Only send extra_body for vLLM (disable thinking mode)
    backend = ncfg.get("backend", "vllm")
    extra_body = {"chat_template_kwargs": {"enable_thinking": False}} if backend == "vllm" else None

    def _nugget_done(paper_id):
        """Check if a paper has already been successfully extracted."""
        path = os.path.join(nugget_dir, f"{paper_id}.json")
        if not os.path.exists(path):
            return False
        try:
            with open(path) as f:
                data = json.load(f)
            return data.get("num_nuggets", 0) > 0
        except (json.JSONDecodeError, OSError):
            return False

    chunk_files = sorted(f for f in os.listdir(chunk_dir) if f.endswith(".json"))
    to_process = []
    skipped_count = 0
    for fname in chunk_files:
        paper_id = fname.replace(".json", "")
        if _nugget_done(paper_id):
            skipped_count += 1
        else:
            to_process.append(fname)

    temp = ext_cfg.get("temperature", 0.1)
    max_tok = ext_cfg.get("max_tokens", 3000)
    max_retries = ext_cfg.get("max_retries", 3)
    retry_delay = ext_cfg.get("retry_base_delay", 2.0)

    print(f"[nuggets] Processing {len(to_process)} papers ({skipped_count} skipped) via {model}, workers={max_workers}")

    # Pipeline multiple papers concurrently. All chunks from all in-flight
    # papers share the same thread pool, keeping the GPU saturated.
    # Each paper is saved to disk as soon as ALL its chunks complete,
    # so a killed job loses at most the in-flight papers (not all of them).
    total_nuggets = 0
    success, failed = 0, 0

    # Load all paper chunks up front (cheap — just JSON metadata)
    paper_chunks = {}
    for fname in to_process:
        paper_id = fname.replace(".json", "")
        try:
            paper_data = load_json(os.path.join(chunk_dir, fname))
            chunks = [c for c in paper_data["chunks"] if len(c["text"].strip()) >= 50]
            if chunks:
                paper_chunks[paper_id] = chunks
            else:
                # No valid chunks — write empty result
                save_json({"paper_id": paper_id, "num_nuggets": 0, "nuggets": []},
                          os.path.join(nugget_dir, f"{paper_id}.json"))
                success += 1
        except Exception as e:
            print(f"  ERROR loading chunks for {paper_id}: {e}")
            failed += 1

    total_papers = len(paper_chunks)
    total_chunks_all = sum(len(c) for c in paper_chunks.values())
    papers_done = 0
    chunks_done = 0
    print_lock = threading.Lock()
    t_start = time.time()
    # Track what each worker is doing: worker_id -> status string
    worker_status = {}

    def _print_status(msg=None):
        """Print a status line showing progress and active workers."""
        elapsed = time.time() - t_start
        mins, secs = divmod(int(elapsed), 60)
        rate = chunks_done / elapsed * 60 if elapsed > 0 else 0
        if msg:
            # Clear line + print message
            sys.stderr.write(f"\r\033[K{msg}\n")
        # Build compact worker activity string
        active = [f"w{k}:{v}" for k, v in sorted(worker_status.items())]
        active_str = " | ".join(active) if active else "idle"
        status = (f"\r\033[K[{papers_done}/{total_papers} papers, "
                  f"{chunks_done}/{total_chunks_all} chunks, "
                  f"{mins:02d}:{secs:02d}, {rate:.1f} chunks/min] {active_str}")
        sys.stderr.write(status)
        sys.stderr.flush()

    def _process_paper(paper_id, chunks, worker_id):
        """Process all chunks of a paper sequentially, passing prior nuggets as context."""
        nonlocal total_nuggets, success, papers_done, chunks_done
        all_nuggets = []
        prior_questions = []
        warnings = []
        n_chunks = len(chunks)
        short_id = paper_id[:25]

        for ci, c in enumerate(chunks):
            with print_lock:
                worker_status[worker_id] = f"{short_id} {ci+1}/{n_chunks}"
                _print_status()
            try:
                nuggets = _process_chunk(
                    client, c, model, temp, max_tok,
                    max_retries, retry_delay, paper_id, extra_body,
                    prior_questions=prior_questions if prior_questions else None)
            except Exception as e:
                nuggets = []
                warnings.append(f"  WARN chunk {c.get('chunk_id', '?')} of {paper_id}: {e}")
            all_nuggets.extend(nuggets)
            prior_questions.extend(n["question"] for n in nuggets)
            with print_lock:
                chunks_done += 1
                total_nuggets += len(nuggets)
                _print_status(f"    {short_id} chunk {ci+1}/{n_chunks} -> {len(nuggets)} nuggets ({len(all_nuggets)} total)")

        deduped = _deduplicate_nuggets(all_nuggets, paper_id)
        output = {
            "paper_id": paper_id,
            "num_nuggets": len(deduped),
            "nuggets": deduped,
        }
        save_json(output, os.path.join(nugget_dir, f"{paper_id}.json"))

        with print_lock:
            # total_nuggets already incremented per-chunk; adjust for dedup
            total_nuggets -= (len(all_nuggets) - len(deduped))
            success += 1
            papers_done += 1
            worker_status.pop(worker_id, None)
            for w in warnings:
                _print_status(w)
            _print_status(f"  ✓ [{papers_done}/{total_papers}] {paper_id[:40]:40s} -> {len(deduped)} nuggets (deduped from {len(all_nuggets)})")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for worker_id, (pid, chunks) in enumerate(paper_chunks.items()):
            futures[executor.submit(_process_paper, pid, chunks, worker_id % max_workers)] = pid
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                pid = futures[fut]
                with print_lock:
                    _print_status(f"  ERROR processing {pid}: {e}")
                failed += 1

    sys.stderr.write("\r\033[K")  # clear status line
    print(f"\nDone: {success} papers, {total_nuggets} nuggets, {failed} failed, {skipped_count} skipped")


def main():
    ap = argparse.ArgumentParser(description="Extract nuggets from chunks")
    ap.add_argument("-c", "--config", default="config.yaml")
    args = ap.parse_args()
    run_extraction(args.config)


if __name__ == "__main__":
    main()
