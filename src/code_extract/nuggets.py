"""LLM-based nugget extraction from code chunks.

Mirrors the paper nugget extraction pipeline (src/nuggets/extract.py)
but uses a code-specific system prompt and produces nugget types:
implementation, config, experiment.
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils import load_config, load_json, save_json, make_llm_client

CODE_PAPER_ID = "__smcm_mcfnet__"

SYSTEM_PROMPT = """You are a code knowledge extractor for a master's thesis on "Spiking Motion Compensation for Energy-Efficient RGB-Event Object Detection." The thesis codebase (SMCM-MCFNet) implements a PyTorch object detection system that fuses RGB images with event camera data using spiking neural networks.

Given a structured summary of a Python module or YAML config, extract implementation knowledge as question-answer nuggets.

Context about the codebase:
- Motion Compensation Module (MCM): estimates optical flow from event voxels and warps events to reduce motion blur
- Spiking Neural Network blocks: LIF/PLIF neurons with BISNN training (ANN pre-train then SNN fine-tune)
- Event Dynamic Upsampling Module (EDUM): bridges low-res event features to high-res RGB features
- Dual-stream backbone: CSPDarknet processing RGB and events separately before fusion
- Optical Flow Estimators (OFE): ST-FlowNet, IDNet, CNNBaseline, ConvFlowNet, etc.
- Detection heads: YOLOX and DETR-style
- Dataset: DSEC-Detection (driving scenes with DVS events + RGB frames)
- Training: distributed PyTorch with iteration-based scheduling, WandB logging

Rules:
- Extract ARCHITECTURAL DECISIONS: why components are structured this way, what design choices were made
- Extract COMPONENT RELATIONSHIPS: how modules connect, what data flows between them
- Extract KEY PARAMETERS: important hyperparameters, their defaults, and what they control
- Extract TRAINING DETAILS: optimizer settings, scheduling, loss functions, batch sizes
- Extract IMPLEMENTATION SPECIFICS: tensor shapes, data representations, GPU memory strategies
- Make each nugget SELF-CONTAINED: a reader should understand it without seeing the code
- Include class/method names naturally: "The MotionCompensationModule uses..." not "The module uses..."
- For config nuggets, explain WHAT the config controls and WHY values were chosen (when inferable)
- Classify type as one of: implementation, config, experiment
  - implementation: architecture, design patterns, component structure, forward pass logic
  - config: hyperparameters, training settings, dataset configs
  - experiment: training run setups, ablation configurations, evaluation settings
- Aim for 3-6 nuggets per chunk. Quality over quantity.
- Output ONLY valid JSON array, no markdown fences, no preamble

Output format:
[
  {
    "question": "How does the MotionCompensationModule combine flow estimation and event warping?",
    "answer": "The MCM composes three sub-modules: (1) a FlowEstimator that predicts per-pixel optical flow from event voxel grids, (2) an EventWarpingModule that bilinearly warps each voxel time-bin to the reference timestamp using the predicted flow, and (3) a loss function (event warping contrast loss or supervised L1 loss). The flow is scaled by flow_scaling (default 640.0, matching MCFNet conventions) before warping. The forward pass returns both the motion-compensated voxel and the MCM loss.",
    "type": "implementation"
  },
  {
    "question": "What are the default training hyperparameters for ANN pre-training of the MCM?",
    "answer": "The mcm_s_ann config uses batch_size=16 per GPU across 4 GPUs (effective 64), SGD optimizer with lr=0.01, cosine annealing schedule over 300 epochs, weight_decay=5e-4, warmup of 5 epochs. MCM loss weight is 0.1 with flow_scaling=640.0 and event_warping contrast loss.",
    "type": "config"
  }
]"""


def repair_json(text):
    """Attempt to extract valid JSON from potentially malformed LLM output."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


def extract_nuggets_from_chunk(client, chunk_text, model, temperature=0.1,
                                max_tokens=2000, extra_body=None,
                                max_model_len=4096):
    """Send a code chunk to the LLM and parse nuggets."""
    try:
        user_content = (
            "Extract implementation knowledge nuggets from this code module summary:\n\n"
            + chunk_text
        )
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

        valid = []
        allowed_types = {"implementation", "config", "experiment"}
        for n in nuggets:
            if isinstance(n, dict) and "question" in n and "answer" in n:
                q = str(n["question"]).strip()
                a = str(n["answer"]).strip()
                if len(q) < 10 or len(a) < 10:
                    continue
                ntype = str(n.get("type", "implementation"))
                if ntype not in allowed_types:
                    ntype = "implementation"
                valid.append({
                    "question": q,
                    "answer": a,
                    "type": ntype,
                })
        return valid, raw
    except Exception as e:
        return [], str(e)


def _process_chunk(client, chunk, model, temp, max_tok, max_retries,
                   retry_delay, extra_body=None, max_model_len=4096):
    """Process a single code chunk with retries."""
    text = chunk["chunk_text"]
    if len(text.strip()) < 50:
        return []

    for attempt in range(max_retries):
        nuggets, raw = extract_nuggets_from_chunk(
            client, text, model, temp, max_tok,
            extra_body=extra_body, max_model_len=max_model_len,
        )
        if nuggets:
            break
        raw_str = str(raw).lower()
        if "rate" in raw_str:
            time.sleep(retry_delay * (2 ** attempt))
        else:
            break

    source_file = chunk.get("file", "")
    for n in nuggets:
        n["paper_id"] = CODE_PAPER_ID
        n["source_file"] = source_file
        n["section"] = "code"
        n["source_chunk"] = chunk.get("module", source_file)
    return nuggets


def _deduplicate_nuggets(all_nuggets):
    """Deduplicate nuggets (>60% word overlap)."""
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
        n["nugget_id"] = f"{CODE_PAPER_ID}_{i}"
    return deduped


def run_extraction(config_path="config.yaml"):
    """Extract nuggets from all code and config chunks."""
    cfg = load_config(config_path)
    chunk_dir = cfg["paths"]["code_chunk_dir"]
    nugget_dir = cfg["paths"]["code_nugget_dir"]
    os.makedirs(nugget_dir, exist_ok=True)

    nug_cfg = cfg.get("nuggets", {})
    ext_cfg = nug_cfg.get("extraction", {})
    backend = nug_cfg.get("backend", "vllm")

    client, model = make_llm_client(cfg)
    vllm_cfg = nug_cfg.get("vllm", {})
    max_model_len = vllm_cfg.get("max_model_len", 8192) if backend == "vllm" else 8192
    max_workers = ext_cfg.get("max_workers", 4)
    max_retries = ext_cfg.get("max_retries", 3)
    max_tokens = ext_cfg.get("max_tokens", 1500)
    temperature = ext_cfg.get("temperature", 0.1)
    retry_delay = ext_cfg.get("retry_base_delay", 2.0)

    extra_body = None
    if backend == "vllm":
        extra_body = {"guided_json": None}  # no constrained decoding

    # Load code chunks + config chunks
    all_chunks = []
    code_path = os.path.join(chunk_dir, "code_chunks.json")
    config_path_file = os.path.join(chunk_dir, "config_chunks.json")
    if os.path.exists(code_path):
        all_chunks.extend(load_json(code_path))
    if os.path.exists(config_path_file):
        all_chunks.extend(load_json(config_path_file))

    print(f"[code_nuggets] {len(all_chunks)} chunks to process")

    all_nuggets = []
    done = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _process_chunk, client, chunk, model, temperature,
                max_tokens, max_retries, retry_delay,
                extra_body=extra_body, max_model_len=max_model_len,
            ): chunk
            for chunk in all_chunks
        }
        for future in as_completed(futures):
            chunk = futures[future]
            try:
                nuggets = future.result()
                all_nuggets.extend(nuggets)
            except Exception as e:
                print(f"  ERROR processing {chunk.get('file', '?')}: {e}")
            done += 1
            if done % 10 == 0 or done == len(all_chunks):
                print(f"  {done}/{len(all_chunks)} chunks done, {len(all_nuggets)} nuggets so far")

    # Deduplicate
    deduped = _deduplicate_nuggets(all_nuggets)
    print(f"\n  {len(all_nuggets)} raw -> {len(deduped)} after dedup")

    # Save
    out_path = os.path.join(nugget_dir, f"{CODE_PAPER_ID}.json")
    save_json(deduped, out_path)
    print(f"  Saved to {out_path}")

    # Also save in the format expected by the embed stage
    # (list of nuggets with all fields)
    return deduped


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Extract nuggets from code chunks")
    ap.add_argument("-c", "--config", default="config.yaml")
    args = ap.parse_args()
    run_extraction(args.config)


if __name__ == "__main__":
    main()
