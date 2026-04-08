"""Embed nuggets using Qwen3-Embedding-8B via vLLM."""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import tiktoken
from openai import OpenAI

from src.utils import load_config, load_json, load_jsonl, save_json, save_jsonl


def load_all_nuggets(nugget_dir, augmented_dir=None, unified_dir=None):
    """Load all nuggets, preferring unified JSONL output, then legacy nugget_dir.

    Priority per paper:
    1. unified_dir (.jsonl, already quality-filtered + augmented)
    2. nugget_dir (.jsonl, raw extraction)

    Legacy augmented_dir (.json envelope) is still supported for backwards
    compatibility but is expected to be unused in the active pipeline.
    """
    all_nuggets = []

    # Load unified papers first (already filtered + augmented)
    unified_papers = set()
    if unified_dir and os.path.isdir(unified_dir):
        for fname in sorted(os.listdir(unified_dir)):
            if not fname.endswith(".jsonl"):
                continue
            paper_id = fname.replace(".jsonl", "")
            nuggets = [n for n in load_jsonl(os.path.join(unified_dir, fname))
                       if not n.get("_removed")]
            all_nuggets.extend(nuggets)
            unified_papers.add(paper_id)

    # Load remaining papers from nugget_dir (skip unified ones)
    for fname in sorted(os.listdir(nugget_dir)) if os.path.isdir(nugget_dir) else []:
        if not fname.endswith(".jsonl"):
            continue
        paper_id = fname.replace(".jsonl", "")
        if paper_id in unified_papers:
            continue
        nuggets = [n for n in load_jsonl(os.path.join(nugget_dir, fname))
                   if not n.get("_removed")]

        # Merge augmented data if available (legacy .json envelope format)
        if augmented_dir:
            aug_path = os.path.join(augmented_dir, f"{paper_id}.json")
            if os.path.exists(aug_path):
                aug_data = load_json(aug_path)

                # Build replacement map: original_nugget_id -> improved nugget
                replacements = {}
                for imp in aug_data.get("improved", []):
                    orig_id = imp.get("original_nugget_id")
                    if orig_id:
                        replacements[orig_id] = imp

                # Replace improved nuggets in-place
                for i, n in enumerate(nuggets):
                    nid = n.get("nugget_id", "")
                    if nid in replacements:
                        rep = replacements[nid]
                        # Carry over metadata from original
                        rep["paper_id"] = n.get("paper_id", "")
                        rep["nugget_id"] = nid
                        if "paper_title" in n:
                            rep["paper_title"] = n["paper_title"]
                        if "paper_authors" in n:
                            rep["paper_authors"] = n["paper_authors"]
                        if "paper_year" in n:
                            rep["paper_year"] = n["paper_year"]
                        nuggets[i] = rep

                # Append gap-filled nuggets
                for gf in aug_data.get("gap_filled", []):
                    gf["paper_id"] = paper_id
                    # Copy paper metadata from first nugget if available
                    if nuggets:
                        for key in ("paper_title", "paper_authors", "paper_year"):
                            if key in nuggets[0] and key not in gf:
                                gf[key] = nuggets[0][key]
                    nuggets.append(gf)

        all_nuggets.extend(nuggets)
    return all_nuggets


_tokenizer = tiktoken.get_encoding("cl100k_base")


def format_nugget_text(nugget, instruction, max_tokens=None, mode="query"):
    """Format a nugget for instruction-aware embedding.

    Args:
        nugget: Dict with 'question' and 'answer' keys.
        instruction: Default instruction (used if mode-specific not available).
        max_tokens: Optional token limit for truncation.
        mode: 'query' for search queries, 'document' for nugget indexing.

    In 'document' mode, nugget type and section are prepended as structured
    tags so that vector search can leverage metadata signals.
    In 'query' mode, no metadata is prepended (users search by content, not
    by type/section).
    """
    q = nugget.get("question", "")
    a = nugget.get("answer", "")

    if mode == "document":
        # Include metadata for richer document embeddings
        ntype = nugget.get("type", "")
        section = nugget.get("section", "")
        prefix = ""
        if ntype:
            prefix += f"[{ntype}] "
        if section:
            prefix += f"[{section}] "
        text = f"{prefix}Q: {q} A: {a}"
    else:
        text = f"Q: {q} A: {a}"

    if instruction:
        text = f"Instruct: {instruction}\nQuery: {text}"
    if max_tokens:
        # Apply 5% safety margin: tiktoken (cl100k_base) and the model tokenizer
        # may count tokens differently, so truncate slightly below the limit.
        safe_limit = int(max_tokens * 0.95)
        tokens = _tokenizer.encode(text)
        if len(tokens) > safe_limit:
            import logging

            logging.getLogger("embed").debug(
                "Truncated nugget from %d to %d tokens (paper=%s)",
                len(tokens),
                safe_limit,
                nugget.get("paper_id", "?"),
            )
            text = _tokenizer.decode(tokens[:safe_limit])
    return text


def embed_batch(client, texts, model, dimensions=None):
    """Embed a batch of texts via vLLM."""
    kwargs = {"model": model, "input": texts}
    if dimensions:
        kwargs["dimensions"] = dimensions
    resp = client.embeddings.create(**kwargs)
    return [item.embedding for item in resp.data]


def make_embed_client(cfg):
    """Create OpenAI client and resolve model name from embed config."""
    ecfg = cfg.get("embed", {})
    backend = ecfg.get("backend", "vllm")
    if backend == "ollama":
        ollama_cfg = ecfg.get("ollama", {})
        base_url = ollama_cfg.get("base_url", "http://localhost:11434/v1")
        model = ollama_cfg.get("model", "qwen3-embedding:8b")
    elif backend == "openrouter":
        or_cfg = ecfg.get("openrouter", {})
        base_url = or_cfg.get("base_url", "https://openrouter.ai/api/v1")
        model = or_cfg.get("model", "qwen/qwen3-embedding-8b")
        api_key = os.environ.get(or_cfg.get("api_key_env", "OPENROUTER_API_KEY"), "")
        client = OpenAI(base_url=base_url, api_key=api_key)
        return client, model
    else:
        vllm_cfg = ecfg.get("vllm", {})
        port = int(os.environ.get("VLLM_PORT", vllm_cfg.get("port", 8000)))
        base_url = f"http://localhost:{port}/v1"
        model = vllm_cfg.get("model", "Qwen/Qwen3-Embedding-8B")
    client = OpenAI(base_url=base_url, api_key="none")
    return client, model


def make_embed_clients(cfg):
    """Create multiple OpenAI clients for multi-instance vLLM embedding.

    If VLLM_PORTS env var is set, returns one client per port.
    Otherwise falls back to a single client.
    Multi-instance mode returns a HealthAwareClients wrapper that skips dead instances.

    Returns (clients_list, model) tuple.
    """
    from src.utils import HealthAwareClients

    ports_env = os.environ.get("VLLM_PORTS", "")
    if not ports_env:
        client, model = make_embed_client(cfg)
        return [client], model

    ecfg = cfg.get("embed", {})
    vllm_cfg = ecfg.get("vllm", {})
    model = vllm_cfg.get("model", "Qwen/Qwen3-Embedding-8B")
    ports = [int(p.strip()) for p in ports_env.split(",") if p.strip()]
    clients = [
        OpenAI(base_url=f"http://localhost:{p}/v1", api_key="none") for p in ports
    ]
    return HealthAwareClients(clients, ports), model


def _load_existing_kb(kb_dir):
    """Load existing embedded nuggets and embeddings for incremental mode.

    Returns (existing_nuggets, existing_embeddings) or (None, None) if not found.
    """
    nug_path = os.path.join(kb_dir, "nuggets_with_embeddings.jsonl")
    npy_path = os.path.join(kb_dir, "embeddings.npy")
    if not os.path.exists(nug_path) or not os.path.exists(npy_path):
        return None, None
    existing_nuggets = load_jsonl(nug_path)
    existing_embeddings = np.load(npy_path)
    if len(existing_nuggets) != existing_embeddings.shape[0]:
        print(
            f"[embed] WARNING: nugget count ({len(existing_nuggets)}) != "
            f"embedding rows ({existing_embeddings.shape[0]}), falling back to full rebuild"
        )
        return None, None
    return existing_nuggets, existing_embeddings


def run_embedding(config_path="config.yaml", incremental=False):
    """Embed all nuggets and save to KB directory.

    Args:
        config_path: Path to YAML config.
        incremental: If True, only embed nuggets not already in the KB.
            Existing embeddings are preserved and new ones are appended.
    """
    cfg = load_config(config_path)
    nugget_dir = cfg["paths"]["nugget_dir"]
    augmented_dir = cfg["paths"].get("augmented_dir")
    unified_dir = cfg["paths"].get("unified_dir")
    kb_dir = cfg["paths"]["kb_dir"]
    ecfg = cfg.get("embed", {})
    emb_cfg = ecfg.get("embedding", {})
    os.makedirs(kb_dir, exist_ok=True)

    clients, model = make_embed_clients(cfg)
    num_instances = len(clients)
    print(f"[embed] vLLM instances: {num_instances}")
    batch_size = emb_cfg.get("batch_size", 64)
    dimensions = emb_cfg.get("dimensions", None)
    # Use doc_instruction for indexing if available, fallback to generic instruction
    instruction = emb_cfg.get("doc_instruction", emb_cfg.get("instruction", ""))

    # Get backend-specific max_model_len for truncation
    backend = ecfg.get("backend", "vllm")
    if backend == "ollama":
        max_tokens = ecfg.get("ollama", {}).get("max_model_len", None)
    elif backend == "openrouter":
        max_tokens = ecfg.get("openrouter", {}).get("max_model_len", None)
    else:
        max_tokens = ecfg.get("vllm", {}).get("max_model_len", None)

    # Load nuggets (preferring unified > augmented > raw)
    print("[embed] Loading nuggets...")
    if unified_dir and os.path.isdir(unified_dir):
        print(f"  Loading unified nuggets from {unified_dir}")
        nuggets = load_all_nuggets(nugget_dir, augmented_dir, unified_dir)
    elif augmented_dir and os.path.isdir(augmented_dir):
        print(f"  Merging augmented nuggets from {augmented_dir}")
        nuggets = load_all_nuggets(nugget_dir, augmented_dir)
    else:
        nuggets = load_all_nuggets(nugget_dir)
    print(f"  Loaded {len(nuggets)} nuggets")
    if not nuggets:
        print("No nuggets found.")
        return

    # Incremental mode: reuse existing embeddings, only embed new nuggets
    if incremental:
        existing_nuggets, existing_embeddings = _load_existing_kb(kb_dir)
        if existing_nuggets is not None:
            # Build map from nugget_id -> (embedding row, content key) for existing nuggets
            existing_emb_map = {}
            for n in existing_nuggets:
                idx = n.get("embedding_idx")
                if idx is None or idx >= existing_embeddings.shape[0]:
                    continue
                content_key = f"{n.get('question', '')}:{n.get('answer', '')}"
                existing_emb_map[n["nugget_id"]] = (existing_embeddings[idx], content_key)

            # Split into already-embedded (with same content) and new/changed
            reused_nuggets = []
            reused_embeddings = []
            new_nuggets = []
            for n in nuggets:
                nid = n.get("nugget_id", "")
                if nid in existing_emb_map:
                    emb, old_content = existing_emb_map[nid]
                    new_content = f"{n.get('question', '')}:{n.get('answer', '')}"
                    if new_content == old_content:
                        reused_nuggets.append(n)
                        reused_embeddings.append(emb)
                    else:
                        new_nuggets.append(n)  # content changed, re-embed
                else:
                    new_nuggets.append(n)

            if not new_nuggets:
                print("[embed] No new nuggets to embed, KB is up to date.")
                return

            print(
                f"[embed] Incremental: {len(reused_nuggets)} existing, "
                f"{len(new_nuggets)} new to embed"
            )
            nuggets_to_embed = new_nuggets
        else:
            print("[embed] No existing KB found, doing full embedding")
            incremental = False

    if not incremental:
        nuggets_to_embed = nuggets

    # Format texts (truncate to max_model_len if configured)
    # Use mode="document" to include type/section metadata in embeddings
    texts = [
        format_nugget_text(n, instruction, max_tokens=max_tokens, mode="document")
        for n in nuggets_to_embed
    ]

    # Embed in batches with pipelined submission
    embed_workers = emb_cfg.get("embed_workers", 4)
    print(
        f"[embed] Embedding {len(texts)} nuggets (batch_size={batch_size}, workers={embed_workers})..."
    )

    # Build batch ranges
    batch_ranges = [
        (s, min(s + batch_size, len(texts))) for s in range(0, len(texts), batch_size)
    ]
    total_batches = len(batch_ranges)

    # Results indexed by batch order to maintain alignment
    results_by_idx = [None] * total_batches

    max_retries = emb_cfg.get("max_retries", 3)

    def _embed_one_batch(idx, start, end):
        client = clients[idx % num_instances]
        for attempt in range(max_retries):
            try:
                embs = embed_batch(client, texts[start:end], model, dimensions)
                return idx, embs
            except Exception as e:
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)
                    print(f"  RETRY batch {idx} attempt {attempt+1}/{max_retries}: {e}")
                else:
                    raise

    failed_batches = []
    with ThreadPoolExecutor(max_workers=embed_workers) as pool:
        futures = {
            pool.submit(_embed_one_batch, i, s, e): i
            for i, (s, e) in enumerate(batch_ranges)
        }
        done = 0
        for fut in as_completed(futures):
            batch_idx = futures[fut]
            try:
                idx, embs = fut.result()
                results_by_idx[idx] = embs
            except Exception as e:
                print(f"  ERROR batch {batch_idx}: {e}")
                failed_batches.append(batch_idx)
            done += 1
            if done % 10 == 0 or done == total_batches:
                print(f"  {done}/{total_batches} batches embedded")

    if failed_batches:
        raise RuntimeError(
            f"{len(failed_batches)} of {total_batches} embedding batches failed "
            f"(batch indices: {failed_batches[:20]}). "
            f"No partial output saved to avoid misaligned embeddings."
        )

    new_embeddings = []
    for embs in results_by_idx:
        new_embeddings.extend(embs)

    # Combine existing + new embeddings in incremental mode
    if incremental and existing_nuggets is not None:
        all_nuggets = reused_nuggets + new_nuggets
        all_embeddings = reused_embeddings + new_embeddings
    else:
        all_nuggets = nuggets
        all_embeddings = new_embeddings

    # Save embeddings as numpy matrix
    emb_matrix = np.array(all_embeddings, dtype=np.float32)
    npy_path = os.path.join(kb_dir, "embeddings.npy")
    np.save(npy_path, emb_matrix)
    print(f"  Saved {npy_path}: shape={emb_matrix.shape}")

    # Save nuggets with embedding index
    for i, nugget in enumerate(all_nuggets):
        nugget["embedding_idx"] = i
    nug_path = os.path.join(kb_dir, "nuggets_with_embeddings.jsonl")
    save_jsonl(all_nuggets, nug_path)
    print(f"  Saved {nug_path}: {len(all_nuggets)} nuggets")


def main():
    ap = argparse.ArgumentParser(description="Embed nuggets")
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument(
        "--incremental",
        action="store_true",
        help="Only embed new nuggets not already in the KB",
    )
    args = ap.parse_args()
    run_embedding(args.config, incremental=args.incremental)


if __name__ == "__main__":
    main()
