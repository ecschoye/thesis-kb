import json, yaml, os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def load_config(path="config.yaml"):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # Expand ~ in all path values
    for key, val in cfg.get("paths", {}).items():
        if isinstance(val, str) and "~" in val:
            cfg["paths"][key] = os.path.expanduser(val)
    return cfg

def load_json(path):
    with open(path) as f:
        return json.load(f)

def save_json(data, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

def paper_id_from_path(path):
    return Path(path).stem

def make_llm_client(cfg):
    """Create an OpenAI-compatible LLM client from nuggets config.

    Returns (client, model) tuple.
    """
    ncfg = cfg.get("nuggets", {})
    backend = ncfg.get("backend", "vllm")
    if backend == "ollama":
        ollama_cfg = ncfg.get("ollama", {})
        base_url = ollama_cfg.get("base_url", "http://127.0.0.1:11434/v1")
        model = ollama_cfg.get("model", "qwen3.5:27b")
        client = OpenAI(base_url=base_url, api_key="ollama")
    elif backend == "openrouter":
        or_cfg = ncfg.get("openrouter", {})
        base_url = or_cfg.get("base_url", "https://openrouter.ai/api/v1")
        model = or_cfg.get("model", "qwen/qwen3.5-32b")
        api_key = os.environ.get(or_cfg.get("api_key_env", "OPENROUTER_API_KEY"), "")
        client = OpenAI(base_url=base_url, api_key=api_key)
    else:
        vllm_cfg = ncfg.get("vllm", {})
        port = int(os.environ.get("VLLM_PORT", vllm_cfg.get("port", 8000)))
        model = vllm_cfg.get("model", "Qwen/Qwen3.5-27B")
        client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="none")
    return client, model


def already_processed(paper_id, output_dir, ext=".json"):
    path = os.path.join(output_dir, f"{paper_id}{ext}")
    if not os.path.exists(path):
        return False
    try:
        return os.path.getsize(path) > 2
    except OSError:
        return False
