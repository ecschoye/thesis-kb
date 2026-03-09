"""LLM-based nugget extraction via vLLM OpenAI-compatible API."""
import os, json, re, time, argparse, threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from src.utils import load_config, load_json, save_json


SYSTEM_PROMPT = """You are a research knowledge extractor for a thesis on RGB-Event camera fusion for object detection on resource-constrained autonomous platforms. The thesis covers: spiking neural networks (SNNs), event camera processing, RGB-Event fusion, optical flow estimation, motion compensation, object detection, short-term trajectory prediction, neuromorphic/energy-efficient hardware, and low-latency inference.

Given a chunk of an academic paper, extract the important factual claims as atomic question-answer nuggets.

Rules:
- Extract ALL substantive contributions of the paper (methods, results, comparisons, limitations, key claims)
- Give EXTRA DEPTH to thesis-relevant topics: extract fine-grained details about spike encoding, membrane potentials, event representations (voxel grids, time surfaces, spike tensors), motion compensation, temporal processing, fusion architectures, energy/latency/FPS metrics on embedded hardware, neuromorphic chip results, DVS/DAVIS sensor specifics, optical flow, SNN architectures, trajectory prediction
- For non-thesis topics in the paper, still extract key contributions but at a higher level (fewer nuggets, less granular)
- Skip trivial metadata: paper title, author names, affiliations, section headings, acknowledgments
- Skip generic background that any textbook covers — only extract background claims specific to this work
- Each nugget must be self-contained (understandable without the source text)
- COMBINE related facts into single nuggets rather than splitting them (e.g. one nugget for "FPGA resource usage" covering FFs, LUTs, and BRAM together)
- Include specific numbers: accuracies, FPS, energy, parameters, latency, FLOPs
- Answers should be DETAILED with exact values when available
- Classify type: method, result, claim, limitation, comparison, background
- Skip content that is purely transitional or structural ("The paper is organized as follows...")
- Aim for 3-10 nuggets per chunk. Prefer fewer, higher-quality nuggets over exhaustive coverage.
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


def extract_nuggets_from_chunk(client, chunk_text, model, temperature=0.1, max_tokens=3000):
    """Send a chunk to the LLM and parse nuggets."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Extract the key knowledge nuggets from this academic text:\n\n{chunk_text}"},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
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


def _process_chunk(client, chunk, model, temp, max_tok, max_retries, retry_delay, paper_id):
    """Process a single chunk — designed for use in a thread pool."""
    text = chunk["text"]
    if len(text.strip()) < 50:
        return []

    for attempt in range(max_retries):
        nuggets, raw = extract_nuggets_from_chunk(client, text, model, temp, max_tok)
        if nuggets:
            break
        if "rate" in str(raw).lower():
            time.sleep(retry_delay * (2 ** attempt))
        else:
            break

    for n in nuggets:
        n["paper_id"] = paper_id
        n["source_chunk"] = chunk["chunk_id"]
        n["section"] = chunk.get("section", "unknown")
        n["pages"] = chunk.get("pages", [])
    return nuggets


def _deduplicate_nuggets(all_nuggets, paper_id):
    """Deduplicate nuggets from overlapping chunks (>75% word overlap)."""
    deduped = []
    deduped_wordsets = []
    for n in all_nuggets:
        words = set(n["question"].lower().split()) | set(n["answer"].lower().split())
        is_dup = False
        for kept_words in deduped_wordsets:
            overlap = len(words & kept_words) / max(len(words | kept_words), 1)
            if overlap > 0.75:
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

    # Track per-paper state: how many chunks remain, collected nuggets
    paper_remaining = {}       # paper_id -> number of outstanding futures
    paper_nuggets = defaultdict(list)  # paper_id -> collected nuggets so far
    paper_warnings = defaultdict(list)
    paper_lock = threading.Lock()
    papers_done = 0

    total_papers = len(paper_chunks)

    def _on_chunk_done(fut, paper_id, chunk):
        """Callback: collect result, save paper when all its chunks are done."""
        nonlocal total_nuggets, success, papers_done
        try:
            nuggets = fut.result()
        except Exception as e:
            nuggets = []
            paper_warnings[paper_id].append(
                f"  WARN chunk {chunk.get('chunk_id', '?')} of {paper_id}: {e}")

        with paper_lock:
            paper_nuggets[paper_id].extend(nuggets)
            paper_remaining[paper_id] -= 1

            if paper_remaining[paper_id] == 0:
                # All chunks for this paper are done — dedup and save immediately
                deduped = _deduplicate_nuggets(paper_nuggets[paper_id], paper_id)
                output = {
                    "paper_id": paper_id,
                    "num_nuggets": len(deduped),
                    "nuggets": deduped,
                }
                save_json(output, os.path.join(nugget_dir, f"{paper_id}.json"))
                total_nuggets += len(deduped)
                success += 1
                papers_done += 1
                for w in paper_warnings[paper_id]:
                    print(w)
                print(f"  [{papers_done}/{total_papers}] {paper_id[:40]:40s} -> {len(deduped)} nuggets")
                # Free memory
                del paper_nuggets[paper_id]
                del paper_remaining[paper_id]
                paper_warnings.pop(paper_id, None)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for paper_id, chunks in paper_chunks.items():
            paper_remaining[paper_id] = len(chunks)
            for c in chunks:
                fut = executor.submit(
                    _process_chunk, client, c, model, temp, max_tok,
                    max_retries, retry_delay, paper_id)
                fut.add_done_callback(lambda f, pid=paper_id, ch=c: _on_chunk_done(f, pid, ch))

    print(f"\nDone: {success} papers, {total_nuggets} nuggets, {failed} failed, {skipped_count} skipped")


def main():
    ap = argparse.ArgumentParser(description="Extract nuggets from chunks")
    ap.add_argument("-c", "--config", default="config.yaml")
    args = ap.parse_args()
    run_extraction(args.config)


if __name__ == "__main__":
    main()
