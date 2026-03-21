"""Embed nuggets using Qwen3-Embedding-8B via vLLM."""
import os, argparse, json
import numpy as np
import tiktoken
from openai import OpenAI
from src.utils import load_config, load_json, save_json


def load_all_nuggets(nugget_dir, augmented_dir=None, unified_dir=None):
    """Load all nuggets, preferring unified output, then augmented merge.

    Priority per paper:
    1. unified_dir (already quality-filtered + augmented)
    2. nugget_dir + augmented_dir merge (legacy pipeline)
    3. nugget_dir only (raw extraction)
    """
    all_nuggets = []

    # Load unified papers first (already filtered + augmented)
    unified_papers = set()
    if unified_dir and os.path.isdir(unified_dir):
        for fname in sorted(os.listdir(unified_dir)):
            if not fname.endswith(".json"):
                continue
            data = load_json(os.path.join(unified_dir, fname))
            all_nuggets.extend(data.get("nuggets", []))
            unified_papers.add(fname)

    # Load remaining papers from nugget_dir (skip unified ones)
    for fname in sorted(os.listdir(nugget_dir)):
        if not fname.endswith(".json"):
            continue
        if fname in unified_papers:
            continue
        data = load_json(os.path.join(nugget_dir, fname))
        nuggets = data.get("nuggets", [])

        # Merge augmented data if available
        if augmented_dir:
            aug_path = os.path.join(augmented_dir, fname)
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
                paper_id = data.get("paper_id", fname.replace(".json", ""))
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
    """
    q = nugget.get("question", "")
    a = nugget.get("answer", "")
    text = f"Q: {q} A: {a}"
    if instruction:
        text = f"Instruct: {instruction}\nQuery: {text}"
    if max_tokens:
        tokens = _tokenizer.encode(text)
        if len(tokens) > max_tokens:
            text = _tokenizer.decode(tokens[:max_tokens])
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
        port = vllm_cfg.get("port", 8000)
        base_url = f"http://localhost:{port}/v1"
        model = vllm_cfg.get("model", "Qwen/Qwen3-Embedding-8B")
    client = OpenAI(base_url=base_url, api_key="none")
    return client, model


def run_embedding(config_path="config.yaml"):
    """Embed all nuggets and save to KB directory."""
    cfg = load_config(config_path)
    nugget_dir = cfg["paths"]["nugget_dir"]
    augmented_dir = cfg["paths"].get("augmented_dir")
    unified_dir = cfg["paths"].get("unified_dir")
    kb_dir = cfg["paths"]["kb_dir"]
    ecfg = cfg.get("embed", {})
    emb_cfg = ecfg.get("embedding", {})
    os.makedirs(kb_dir, exist_ok=True)

    client, model = make_embed_client(cfg)
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
        print("No nuggets found."); return

    # Format texts (truncate to max_model_len if configured)
    texts = [format_nugget_text(n, instruction, max_tokens=max_tokens) for n in nuggets]

    # Embed in batches
    print(f"[embed] Embedding {len(texts)} nuggets (batch_size={batch_size})...")
    all_embeddings = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        embs = embed_batch(client, batch, model, dimensions)
        all_embeddings.extend(embs)
        batch_num = start // batch_size + 1
        if batch_num % 10 == 0:
            print(f"  {start + len(batch)}/{len(texts)} embedded")

    # Save embeddings as numpy matrix
    emb_matrix = np.array(all_embeddings, dtype=np.float32)
    npy_path = os.path.join(kb_dir, "embeddings.npy")
    np.save(npy_path, emb_matrix)
    print(f"  Saved {npy_path}: shape={emb_matrix.shape}")

    # Save nuggets with embedding index
    for i, nugget in enumerate(nuggets):
        nugget["embedding_idx"] = i
    nug_path = os.path.join(kb_dir, "nuggets_with_embeddings.json")
    save_json(nuggets, nug_path)
    print(f"  Saved {nug_path}: {len(nuggets)} nuggets")


def main():
    ap = argparse.ArgumentParser(description="Embed nuggets")
    ap.add_argument("-c", "--config", default="config.yaml")
    args = ap.parse_args()
    run_embedding(args.config)


if __name__ == "__main__":
    main()
