---
paths:
  - "src/acquire/**"
  - "src/extract/**"
  - "src/chunk/**"
  - "src/nuggets/**"
  - "src/embed/**"
  - "src/store/**"
---

# Pipeline Rules

- Extract, Chunk, Nuggets stages are **incremental** — they skip papers that already have output files. Always preserve this behavior.
- Embed and Store are **full rebuilds** — they reload all nuggets and recreate the KB each time.
- All paths (pdf_dir, corpus dirs, kb dir) come from config.yaml — never hardcode paths.
- Use `logging.getLogger(__name__)` for all logging — the project uses `src/log.py` with rotating file handlers.
- Paper IDs use underscores (`2401_17151`); arXiv IDs use dots (`2401.17151`). Convert correctly.
- Each pipeline stage is a Python module (`python -m src.<stage>`). Maintain this pattern for new stages.
- To reprocess a paper, delete its output file and rerun — do not add "force" flags.
