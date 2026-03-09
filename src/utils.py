import json, yaml, os
from pathlib import Path

def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

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

def already_processed(paper_id, output_dir, ext=".json"):
    path = os.path.join(output_dir, f"{paper_id}{ext}")
    if not os.path.exists(path):
        return False
    try:
        return os.path.getsize(path) > 2
    except OSError:
        return False
