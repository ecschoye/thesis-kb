import json
import yaml
import os
import time
import threading
import urllib.request
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

def save_jsonl(items, path, removed=None):
    """Write items as JSONL (one JSON object per line), atomically."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        if removed:
            for item in removed:
                item["_removed"] = True
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def load_jsonl(path):
    """Read a JSONL file, returning a list of parsed objects. Skips blank lines."""
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def count_jsonl(path):
    """Count non-empty lines in a JSONL file without parsing."""
    count = 0
    with open(path) as f:
        for line in f:
            if line.strip():
                count += 1
    return count


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
        fallback_models = or_cfg.get("fallback_models", [])
        api_key = os.environ.get(or_cfg.get("api_key_env", "OPENROUTER_API_KEY"), "")
        client = OpenAI(base_url=base_url, api_key=api_key)
        if fallback_models:
            client._fallback_models = fallback_models
    else:
        vllm_cfg = ncfg.get("vllm", {})
        port = int(os.environ.get("VLLM_PORT", vllm_cfg.get("port", 8000)))
        model = vllm_cfg.get("model", "Qwen/Qwen3.5-27B")
        client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="none")
    return client, model


class HealthAwareClients:
    """List-like wrapper that skips unhealthy vLLM instances.

    Periodically checks /health on each port. Round-robin indexing
    transparently skips dead instances so requests only go to live ones.
    If a dead instance comes back (watchdog restarted it), it's re-included.
    """

    def __init__(self, clients, ports, check_interval=15):
        self._clients = clients
        self._ports = ports
        self._healthy = [True] * len(clients)
        self._lock = threading.Lock()
        self._check_interval = check_interval
        self._stop = threading.Event()
        if len(clients) > 1:
            self._thread = threading.Thread(target=self._monitor, daemon=True)
            self._thread.start()

    def _monitor(self):
        while not self._stop.wait(self._check_interval):
            for i, port in enumerate(self._ports):
                try:
                    req = urllib.request.Request(
                        f"http://localhost:{port}/health", method="GET"
                    )
                    urllib.request.urlopen(req, timeout=3)
                    alive = True
                except Exception:
                    alive = False
                with self._lock:
                    if self._healthy[i] != alive:
                        import sys
                        status = "UP" if alive else "DOWN"
                        sys.stderr.write(
                            f"\r\033[K  [health] instance {i} port {port}: {status}\n"
                        )
                        sys.stderr.flush()
                    self._healthy[i] = alive

    def get_healthy_client(self, idx):
        """Return a healthy client, preferring the one at idx but falling back."""
        with self._lock:
            n = len(self._clients)
            # Try the preferred index first, then cycle through others
            for offset in range(n):
                i = (idx + offset) % n
                if self._healthy[i]:
                    return self._clients[i]
            # All down — return preferred and let it fail with a normal error
            return self._clients[idx % n]

    def __getitem__(self, idx):
        return self.get_healthy_client(idx)

    def __len__(self):
        with self._lock:
            return max(1, sum(self._healthy))

    def stop(self):
        self._stop.set()


def make_llm_clients(cfg):
    """Create multiple OpenAI-compatible clients for multi-instance vLLM.

    If VLLM_PORTS env var is set (comma-separated), returns one client per port.
    Otherwise falls back to a single client via make_llm_client.
    Multi-instance mode returns a HealthAwareClients wrapper that skips dead instances.

    Returns (clients_list, model) tuple.
    """
    ports_env = os.environ.get("VLLM_PORTS", "")
    if not ports_env:
        client, model = make_llm_client(cfg)
        return [client], model

    ncfg = cfg.get("nuggets", {})
    vllm_cfg = ncfg.get("vllm", {})
    model = vllm_cfg.get("model", "Qwen/Qwen3.5-27B")
    ports = [int(p.strip()) for p in ports_env.split(",") if p.strip()]
    clients = [OpenAI(base_url=f"http://localhost:{p}/v1", api_key="none") for p in ports]
    return HealthAwareClients(clients, ports), model


def already_processed(paper_id, output_dir, ext=".json"):
    path = os.path.join(output_dir, f"{paper_id}{ext}")
    if not os.path.exists(path):
        return False
    try:
        return os.path.getsize(path) > 2
    except OSError:
        return False
