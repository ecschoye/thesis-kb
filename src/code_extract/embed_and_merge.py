"""Embed code nuggets and merge with existing paper embeddings.

Avoids re-embedding the full 153K paper nuggets — only embeds the
~465 code nuggets and appends them to the existing KB data.
"""

import json
import os
import numpy as np
from src.utils import load_config, load_json, save_json
from src.embed.embedder import make_embed_client, format_nugget_text


def run(config_path="config.yaml"):
    cfg = load_config(config_path)
    kb_dir = cfg["paths"]["kb_dir"]
    code_nugget_dir = cfg["paths"]["code_nugget_dir"]
    ecfg = cfg.get("embed", {})
    emb_cfg = ecfg.get("embedding", {})

    # Load existing KB data
    existing_nug_path = os.path.join(kb_dir, "nuggets_with_embeddings.json")
    existing_emb_path = os.path.join(kb_dir, "embeddings.npy")

    if os.path.exists(existing_nug_path) and os.path.exists(existing_emb_path):
        print("[merge] Loading existing KB data...")
        existing_nuggets = load_json(existing_nug_path)
        existing_embeddings = np.load(existing_emb_path)
        print(f"  {len(existing_nuggets)} existing nuggets, {existing_embeddings.shape}")

        # Remove any previous code nuggets (re-run safe)
        code_ids = set()
        keep_indices = []
        for i, n in enumerate(existing_nuggets):
            if n.get("paper_id") == "__smcm_mcfnet__":
                code_ids.add(n.get("nugget_id"))
            else:
                keep_indices.append(i)

        if code_ids:
            print(f"  Removing {len(code_ids)} previous code nuggets")
            existing_nuggets = [existing_nuggets[i] for i in keep_indices]
            existing_embeddings = existing_embeddings[keep_indices]
    else:
        print("[merge] No existing KB data found — embedding code nuggets only")
        existing_nuggets = []
        existing_embeddings = np.empty((0, 0))

    # Load code nuggets
    code_nugget_file = os.path.join(code_nugget_dir, "__smcm_mcfnet__.json")
    if not os.path.exists(code_nugget_file):
        print(f"ERROR: {code_nugget_file} not found. Run code_extract.nuggets first.")
        return
    code_nuggets = load_json(code_nugget_file)
    print(f"[merge] {len(code_nuggets)} code nuggets to embed")

    # Embed code nuggets
    client, model = make_embed_client(cfg)
    instruction = emb_cfg.get("doc_instruction", emb_cfg.get("instruction", ""))
    batch_size = emb_cfg.get("batch_size", 64)

    # Get max_tokens for truncation
    backend = ecfg.get("backend", "vllm")
    if backend == "ollama":
        max_tokens = ecfg.get("ollama", {}).get("max_model_len", None)
    elif backend == "openrouter":
        max_tokens = ecfg.get("openrouter", {}).get("max_model_len", None)
    else:
        max_tokens = ecfg.get("vllm", {}).get("max_model_len", None)

    texts = [format_nugget_text(n, instruction, max_tokens=max_tokens, mode="document") for n in code_nuggets]

    print(f"[merge] Embedding {len(texts)} code nuggets...")
    all_embeddings = []
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch = texts[start:end]
        resp = client.embeddings.create(model=model, input=batch)
        batch_embs = [d.embedding for d in resp.data]
        all_embeddings.extend(batch_embs)
        print(f"  {end}/{len(texts)} embedded")

    code_embeddings = np.array(all_embeddings, dtype=np.float32)
    print(f"  Code embeddings shape: {code_embeddings.shape}")

    # Merge
    if existing_embeddings.size > 0:
        merged_nuggets = existing_nuggets + code_nuggets
        merged_embeddings = np.vstack([existing_embeddings, code_embeddings])
    else:
        merged_nuggets = code_nuggets
        merged_embeddings = code_embeddings

    print(f"[merge] Merged: {len(merged_nuggets)} nuggets, {merged_embeddings.shape}")

    # Save
    save_json(merged_nuggets, existing_nug_path)
    np.save(existing_emb_path, merged_embeddings)
    print(f"[merge] Saved to {kb_dir}")

    # Rebuild SQLite + ChromaDB
    print("\n[merge] Rebuilding KB...")
    from src.store.kb import run_build
    run_build(config_path)

    print("\nDone! Code nuggets merged into KB.")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Embed code nuggets and merge into KB")
    ap.add_argument("-c", "--config", default="config.yaml")
    args = ap.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
