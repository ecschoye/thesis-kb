# Paper Discovery Tool

Finds new relevant papers for the thesis by searching arXiv, Semantic Scholar, and OpenAlex. Dynamically adapts to your current research focus as you add papers to the KB.

## How It Works

```
KB nuggets + bibliography + local PDFs
        ↓
  Dynamic query profile (15 topic queries + 42 seed papers)
        ↓
  Search APIs (arXiv, S2 keyword, S2 citation graph, OpenAlex)
        ↓
  ~750 raw candidates
        ↓
  Dedup against:
    - KB manifest (arXiv ID, DOI, fuzzy title)
    - ~/Documents/thesis-papers (title from filename)
    - Cross-source duplicates
        ↓
  ~570 new papers
        ↓
  Score (keyword overlap + authority + optional embeddings)
        ↓
  Top N downloaded to ~/Documents/thesis-papers
  JSON report saved to corpus/discovery_reports/
```

## Usage

```bash
cd ~/thesis-kb

# Default: search S2 + arXiv, download top 100
python -m src.discover run

# S2 only (faster, no arXiv rate limit issues)
python -m src.discover run --sources s2

# Just browse, don't download
python -m src.discover run --no-download

# Download top 30 only
python -m src.discover run --download-top 30

# With citation graph expansion (find papers that cite your references)
python -m src.discover run --citation-expand 20

# Recent papers only (last 3 months)
python -m src.discover run --date-from 2026-01-01

# Without embeddings (faster)
python -m src.discover run --no-embeddings

# Full JSON report to stdout
python -m src.discover run --json

# Custom local papers directory
python -m src.discover --local-papers /path/to/pdfs
```

## All Flags

| Flag | Default | Description |
|------|---------|-------------|
| `-n`, `--max-results` | 100 | Top N papers to report + download |
| `--sources` | `arxiv s2` | Which APIs to query (`arxiv`, `s2`, `openalex`) |
| `--date-from` | 6 months ago | Only papers after this date (ISO format) |
| `--no-embeddings` | off | Skip embedding scoring (no Ollama needed) |
| `--no-download` | off | Don't download PDFs, just show report |
| `--download-top` | same as -n | Override how many to download |
| `--citation-expand` | 20 | Seed papers for S2 citation graph traversal |
| `--local-papers` | ~/Documents/thesis-papers | Local PDF dir for dedup |
| `--bib` | ~/TDT4900.../bibliography.bib | Bibliography file for seed extraction |
| `--json` | off | Output full JSON to stdout |
| `-c`, `--config` | config.yaml | thesis-kb config file |

## Dedup Layers

1. **Exact ID** — arXiv ID or DOI matches KB manifest entry
2. **Fuzzy title** — SequenceMatcher >= 0.85 against KB manifest titles
3. **Local PDFs** — title extracted from `Author - Year - Title.pdf` filenames in ~/Documents/thesis-papers (catches papers you have locally but haven't processed into the KB yet)
4. **Cross-source** — same paper found by both arXiv and S2

## Scoring

| Signal | Weight (with embeddings) | Weight (without) |
|--------|--------------------------|-------------------|
| Embedding similarity | 0.50 | — |
| Keyword overlap | 0.30 | 0.60 |
| Authority (citations + recency) | 0.20 | 0.40 |

- **Keyword overlap**: KB-derived topic terms matched against title (2x weight) + abstract
- **Authority**: `log2(1 + citations) / 15` + recency bonus (0.15 for last year) + venue bonus (0.10 for journal/conference)
- **Embedding similarity**: cosine sim of candidate abstract vs query embeddings (needs Ollama with qwen3-embedding:8b)

## Dynamic Profile

The query profile rebuilds from KB state every run:

- Extracts top bigrams from 50k nugget questions (TF-IDF style)
- Papers added in last 30 days get 3x weight → profile shifts as you add papers
- Seed papers for citation expansion come from bibliography + top-cited KB papers
- Stopword list filters out QA-template noise ("specific limitations", "primary motivation", etc.)

## Files

| File | Purpose |
|------|---------|
| `profile.py` | Dynamic query profile from KB + bib + local PDFs |
| `sources.py` | arXiv, S2, OpenAlex API clients with rate limiting |
| `dedup.py` | 4-layer deduplication |
| `scorer.py` | Relevance scoring (3 signals) |
| `report.py` | JSON report + terminal summary |
| `discover.py` | Main orchestrator |
| `__main__.py` | CLI entry point |

## Ledger

Every candidate ever found (not just top N) is saved to `corpus/discovery_ledger.json`. It accumulates across runs and tracks:

- `best_score` — highest relevance score seen across runs
- `times_seen` — how many runs found this paper
- `status` — `candidate` | `downloaded` | `rejected` | `ingested`
- `first_seen` / `last_seen` — date tracking

```bash
# Browse all candidates (sorted by score)
python -m src.discover ledger

# Sort by citations
python -m src.discover ledger --sort citations

# Papers seen in multiple runs (consistently relevant)
python -m src.discover ledger --sort seen

# Only downloaded papers
python -m src.discover ledger --status downloaded

# High-score candidates you haven't downloaded yet
python -m src.discover ledger --min-score 0.5 --status candidate

# Show more
python -m src.discover ledger --limit 100

# Dump full JSON
python -m src.discover ledger --json
```

### Ledger statuses

| Status | Meaning |
|--------|---------|
| `candidate` | Found but not yet downloaded |
| `downloaded` | PDF downloaded to ~/Documents/thesis-papers |
| `rejected` | Manually marked as not relevant (edit the JSON) |
| `ingested` | Processed into thesis-kb (edit the JSON) |

## Output

**Terminal**: ranked list with score, title, authors, year, source, citations

**JSON report** (`corpus/discovery_reports/YYYY-MM-DD.json`): full metadata for top N recommendations including abstracts, URLs, per-signal scores

**Ledger** (`corpus/discovery_ledger.json`): persistent accumulation of ALL candidates across all runs

**Downloads**: PDFs saved to `~/Documents/thesis-papers/` as `Author et al. - Year - Title.pdf`

## Rate Limits

| Source | Delay | Notes |
|--------|-------|-------|
| arXiv | 5s between requests | Can trigger temp bans if hit too fast. Retry with 10/20/30s backoff |
| Semantic Scholar | 1s (free) / 0.15s (API key) | Set `S2_API_KEY` in `.env` for 10x throughput |
| OpenAlex | 0.15s | Free, no key needed |

## Typical Run Times

| Config | Time | Candidates |
|--------|------|------------|
| `--sources s2 --no-embeddings` | ~30s | ~750 |
| `--sources s2 --citation-expand 10` | ~1min | ~1250 |
| `--sources arxiv s2` | ~2min | ~1500 |
| `--sources arxiv s2 openalex` | ~3min | ~2000 |

## Environment

- `S2_API_KEY` in `~/thesis-kb/.env` — 10x faster S2 queries
- Ollama with `qwen3-embedding:8b` — needed for embedding scoring (optional)
