"""LLM-based nugget extraction via vLLM OpenAI-compatible API."""
import os
import json
import re
import sys
import time
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils import load_config, load_json, save_json, make_llm_client


SYSTEM_PROMPT = """You are a research knowledge extractor for a master's thesis on enhancing autonomous vehicle perception through RGB-Event camera fusion with spiking neural networks.

THESIS CONTEXT — the proposed system has four pillars:
1. Spiking Motion Compensation Module (SMCM): SNN-based optical flow (ST-FlowNet) warps asynchronous events to RGB frame timestamps for temporal alignment, replacing CNN optical flow for energy efficiency.
2. Cross-modal Mamba Module (CMM): intermediate fusion at CSPDarkNet layers 3-5 using state-space models (Mamba) to compute adaptive modality importance weights, inspired by MCFNet.
3. Object Detection Head: FPN-based detection on fused features, evaluated via mAP@50 on DSEC and Gen1 benchmarks.
4. Short-Term Trajectory Prediction (STTP): integrated prediction head outputting per-object direction heading from detection features + optical flow, avoiding separate tracking pipelines.

The thesis answers five research questions about: (1) STTP latency impact on hazard detection, (2) latency vs. computational cost trade-offs, (3) SNN readout strategies for dense optical flow, (4) SNN vs. CNN motion compensation efficiency, (5) SMCM temporal resolution robustness in HDR/high-speed scenes.

Given a chunk of an academic paper, extract the important factual claims as atomic question-answer nuggets.

PRIORITY TOPICS — extract at maximum depth:
- Event camera representations: voxel grids, time surfaces, spike tensors, polarity images, event volumes
- Event-to-frame temporal/spatial alignment: warping, motion compensation, optical flow estimation
- RGB-Event fusion architectures: attention mechanisms, gating, cross-modal modules, feature calibration, Mamba/SSM-based fusion
- SNN architectures: LIF neurons, membrane dynamics, surrogate gradients, readout strategies, spike-driven attention
- SNN training: ANN-to-SNN conversion, hybrid SNN-ANN, direct training with STBP/BPTT
- Optical flow: dense estimation (FlowNet, PWC-Net, RAFT), event-based flow (Spike-FlowNet, ST-FlowNet, SFformerFlow)
- Detection architectures: YOLO variants, DETR, anchor-free detectors, detection on event data
- Energy/efficiency metrics: synaptic operations, spike rates, mJ/inference, FLOPs, MACs, latency, FPS on edge hardware (Jetson, neuromorphic chips, FPGAs)
- Trajectory prediction and tracking: ADE/FDE, time-to-collision, motion forecasting
- Benchmark results on: DSEC, Gen1, COCO, KITTI, N-Caltech101, PKU-DVS-Gesture, MVSEC

STANDARD TOPICS — extract key contributions at a higher level:
- General computer vision, transformers, attention not specific to events/fusion
- Pure RGB object detection improvements
- Autonomous driving components unrelated to perception (planning, control)

Rules:
- Extract ALL substantive contributions (methods, results, comparisons, limitations, motivations)
- Each nugget must be SELF-CONTAINED — understandable without the source text
- COMBINE related facts into single nuggets rather than splitting (e.g. one nugget covering all resource usage metrics together)
- Include SPECIFIC NUMBERS: accuracies, FPS, energy, parameters, latency, FLOPs, dataset sizes
- Answers should be DETAILED with exact values when available
- Classify type: method, result, claim, limitation, comparison, background
- Skip trivial metadata (affiliations, section headings, acknowledgments) and generic textbook background
- NEVER repeat information likely covered in a previous chunk. Each nugget must add NEW information. When in doubt, skip it
- For RESULTS and COMPARISONS: always include dataset/benchmark name, baseline methods compared against, and the specific metric (e.g. "mAP@50 on DSEC" not just "mAP")
- For METHOD nuggets: include training details when present (optimizer, LR schedule, batch size, epochs, GPU type, training time)
- Extract ABLATION results: what component was changed/removed and the resulting metric delta
- Extract MOTIVATION: why the authors propose their approach — what gap or limitation of prior work they address. Classify as "claim"
- When the paper title or author names appear, weave them into the answer naturally for citability
- If the chunk contains the paper title, author list, year, or venue: emit ONE "background" nugget with Q: "What is the title and who are the authors of this paper?" Only emit this for the first chunk (introduction/abstract), not later sections
- Do NOT extract nuggets that admit missing information ("specific details are not provided")
- Do NOT extract nuggets about cited/referenced works from bibliography sections. Only extract knowledge about the paper itself
- Aim for 3-8 nuggets per chunk. Fewer, higher-quality nuggets are always better than exhaustive coverage
- Output ONLY valid JSON array, no markdown fences, no preamble

Output format:
[
  {
    "question": "How does ST-FlowNet estimate optical flow from events and what neuron model does it use?",
    "answer": "ST-FlowNet uses a hybrid SNN encoder with LIF neurons (learnable membrane time constant tau_m=2.0, threshold V_th=1.0, soft reset) that processes voxel grid event representations, paired with an ANN decoder for dense flow prediction. It combines STBP training, ANN-to-SNN weight transfer, and BiSNN bridging with ConvGRU temporal recurrence to achieve stable training at scale.",
    "type": "method"
  },
  {
    "question": "How does MCFNet's cross-modal Mamba module fuse RGB and event features?",
    "answer": "MCFNet applies a Cross-modal Mamba Module (CMM) at CSPDarkNet layers 3, 4, and 5. RGB features F_r and event features F_e are concatenated and processed through a state-space model (Mamba) that computes global context and modality importance weights. The weight map adaptively emphasizes the more informative modality at each spatial location, with residual connections preserving the original features.",
    "type": "method"
  },
  {
    "question": "What detection accuracy does the RGB-Event fusion model achieve compared to single-modality baselines on DSEC?",
    "answer": "The fusion model achieves 52.1% mAP@50 on DSEC, outperforming the RGB-only baseline (45.3% mAP@50) by 6.8 percentage points and the event-only baseline (38.7% mAP@50) by 13.4 percentage points, with the largest gains in nighttime and high-speed driving sequences.",
    "type": "comparison"
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


def extract_nuggets_from_chunk(client, chunk_text, model, temperature=0.1, max_tokens=3000, extra_body=None, prior_questions=None, max_model_len=4096):
    """Send a chunk to the LLM and parse nuggets."""
    try:
        user_content = "Extract the key knowledge nuggets from this academic text:\n\n" + chunk_text
        if prior_questions:
            truncated = prior_questions[-20:]  # last 20 to stay in context
            if len(prior_questions) > 20:
                import logging
                logging.getLogger("nuggets.extract").warning(
                    "Prior questions truncated from %d to 20 — dedup context lost", len(prior_questions))
            already = "\n".join(f"- {q}" for q in truncated)
            user_content += f"\n\nNuggets ALREADY EXTRACTED from earlier chunks of this paper (do NOT repeat these):\n{already}"
        # Cap max_tokens to fit within context window
        input_estimate = (len(SYSTEM_PROMPT) + len(user_content)) // 3
        effective_max_tokens = max_tokens
        if input_estimate + max_tokens > max_model_len:
            effective_max_tokens = max(256, max_model_len - input_estimate)
        kwargs = dict(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=temperature,
            max_tokens=effective_max_tokens,
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
                q = str(n["question"]).strip()
                a = str(n["answer"]).strip()
                if len(q) < 10 or len(a) < 10:
                    continue  # skip empty/trivial nuggets
                valid.append({
                    "question": q,
                    "answer": a,
                    "type": str(n.get("type", "background")),
                })
        return valid, raw
    except Exception as e:
        return [], str(e)


_SKIP_SECTIONS = {"references", "acknowledgments", "acknowledgements",
                   "bibliography", "appendix", "appendices"}

def _process_chunk(client, chunk, model, temp, max_tok, max_retries, retry_delay, paper_id, extra_body=None, prior_questions=None, max_model_len=4096):
    """Process a single chunk — designed for use in a thread pool."""
    text = chunk["text"]
    if len(text.strip()) < 50:
        return []
    # Skip sections that never produce useful nuggets (saves API calls)
    section = chunk.get("section", "").lower().strip()
    if section in _SKIP_SECTIONS:
        return []

    # Build model list: primary + fallbacks (if configured on client)
    models_to_try = [model] + getattr(client, "_fallback_models", [])

    cur_prior = prior_questions
    nuggets = []
    for cur_model in models_to_try:
        for attempt in range(max_retries):
            nuggets, raw = extract_nuggets_from_chunk(client, text, cur_model, temp, max_tok, extra_body=extra_body, prior_questions=cur_prior, max_model_len=max_model_len)
            if nuggets:
                break
            raw_str = str(raw).lower()
            if "rate" in raw_str:
                time.sleep(retry_delay * (2 ** attempt))
            elif "too long" in raw_str or "maximum context" in raw_str or "400" in raw_str or "max_model_len" in raw_str or "prompt is too long" in raw_str:
                if cur_prior:
                    cur_prior = cur_prior[-10:] if len(cur_prior) > 10 else None
                else:
                    break
            else:
                break
        if nuggets:
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
    unified_dir = cfg["paths"].get("unified_dir", "corpus/nuggets_unified")
    ncfg = cfg.get("nuggets", {})
    ext_cfg = ncfg.get("extraction", {})
    max_workers = ext_cfg.get("max_workers", 8)
    os.makedirs(nugget_dir, exist_ok=True)

    client, model = make_llm_client(cfg)

    # Only send extra_body for vLLM (disable thinking mode)
    backend = ncfg.get("backend", "vllm")
    extra_body = {"chat_template_kwargs": {"enable_thinking": False}} if backend == "vllm" else None

    def _nugget_done(paper_id):
        """Check if a paper has already been successfully extracted.

        Checks both nugget_dir (raw) and unified_dir to avoid
        re-extracting papers that already went through the unified pipeline.
        """
        for d in (nugget_dir, unified_dir):
            path = os.path.join(d, f"{paper_id}.json")
            if not os.path.exists(path):
                continue
            try:
                with open(path) as f:
                    data = json.load(f)
                if data.get("num_nuggets", 0) > 0:
                    return True
            except (json.JSONDecodeError, OSError):
                continue
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
                save_json({"paper_id": paper_id, "num_nuggets": 0, "num_removed": 0,
                           "num_improved": 0, "num_gap_filled": 0,
                           "quality_summary": {}, "nuggets": [], "removed": []},
                          os.path.join(unified_dir, f"{paper_id}.json"))
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
            "num_removed": 0,
            "num_improved": 0,
            "num_gap_filled": 0,
            "quality_summary": {},
            "nuggets": deduped,
            "removed": [],
        }
        save_json(output, os.path.join(unified_dir, f"{paper_id}.json"))

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
