"""Enrich nugget files with paper metadata (title, authors, year) from manifest."""
import os, json, argparse
from src.utils import load_config, load_json, save_json


def load_manifest(manifest_path):
    """Load manifest keyed by paper_id."""
    papers = {}
    with open(manifest_path) as f:
        manifest = json.load(f)
    for entry in manifest:
        pid = entry.get("paper_id", "")
        if pid:
            papers[pid] = entry
    return papers


def enrich_nuggets(config_path="config.yaml"):
    cfg = load_config(config_path)
    nugget_dir = cfg["paths"]["nugget_dir"]
    manifest_path = os.path.join(cfg["paths"]["corpus_dir"], "manifest.json")
    papers = load_manifest(manifest_path)

    files = sorted(f for f in os.listdir(nugget_dir) if f.endswith(".json"))
    updated = 0

    for fname in files:
        paper_id = fname.replace(".json", "")
        info = papers.get(paper_id, {})
        title = info.get("title", "")
        authors = info.get("authors_str", "")
        year = info.get("year", "")

        path = os.path.join(nugget_dir, fname)
        data = load_json(path)

        changed = False
        for n in data.get("nuggets", []):
            if n.get("paper_title") != title or n.get("paper_authors") != authors or n.get("paper_year") != year:
                n["paper_title"] = title
                n["paper_authors"] = authors
                n["paper_year"] = year
                changed = True

        if changed:
            save_json(data, path)
            updated += 1

    print(f"[enrich] Enriched {updated}/{len(files)} nugget files with paper metadata")


def main():
    ap = argparse.ArgumentParser(description="Enrich nuggets with paper metadata")
    ap.add_argument("-c", "--config", default="config.yaml")
    args = ap.parse_args()
    enrich_nuggets(args.config)


if __name__ == "__main__":
    main()
