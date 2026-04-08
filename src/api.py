"""FastAPI backend for thesis-kb web chat interface."""
import os
import json
import asyncio
import re
import time
import math
import threading
import httpx
from collections import defaultdict
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator
from functools import lru_cache
from openai import OpenAI

from src.utils import load_config
from src.query import ThesisKB
from src.embed.embedder import make_embed_client, format_nugget_text
from src.rerank import rerank_nuggets
from src.log import get_logger

log = get_logger("api", "api.log")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_config_path = os.environ.get("THESIS_KB_CONFIG", "config.yaml")
_cfg = None
_kb: ThesisKB | None = None
_embed_client = None
_embed_model = None
_embed_instruction = ""
_collection = None
_executor = None  # initialized in _init()
_retrieval_cfg: dict = {}  # loaded from config.yaml retrieval section
_bib_lookup: dict[str, str] = {}  # normalised key → bibtex cite key
_feedback_db = None  # separate SQLite connection for feedback
_feedback_lock = threading.Lock()  # serialize feedback DB writes
_trace_lock = threading.Lock()


def _write_retrieval_trace(query, mode, elapsed, variants, effective_config,
                           query_delta, hyde_passage, rrf_scores, nugget_data,
                           overlap_count, top_ids):
    """Write a structured retrieval trace to logs/retrieval_traces.jsonl."""
    try:
        trace_path = os.path.join("logs", "retrieval_traces.jsonl")
        os.makedirs("logs", exist_ok=True)

        # Top-30 by RRF score with key metadata
        ranked = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:30]
        scored_nuggets = []
        for nid in ranked:
            nd = nugget_data.get(nid, {})
            scored_nuggets.append({
                "nugget_id": nid,
                "paper_id": nd.get("paper_id", ""),
                "type": nd.get("type", ""),
                "section": nd.get("section", ""),
                "rrf_score": round(rrf_scores.get(nid, 0), 6),
                "overlap": overlap_count.get(nid, 0),
                "overall_score": nd.get("overall_score"),
                "thesis_relevance": nd.get("thesis_relevance", 0),
            })

        # Compact effective config (drop None values)
        config_compact = {k: v for k, v in effective_config.items() if v is not None}

        trace = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "query": query,
            "mode": mode,
            "elapsed_s": round(elapsed, 3),
            "n_variants": len(variants) if variants else 0,
            "hyde_enabled": query_delta.get("hyde_enabled"),
            "hyde_reason": query_delta.get("hyde_reason", ""),
            "hyde_passage": hyde_passage,
            "bm25_weight": effective_config.get("bm25_weight", 1.0),
            "effective_config": config_compact,
            "query_delta": {k: v for k, v in query_delta.items() if v is not None},
            "n_candidates": len(rrf_scores),
            "n_returned": len(top_ids),
            "top_nuggets": scored_nuggets,
            "returned_ids": list(top_ids[:20]),
        }

        with _trace_lock:
            with open(trace_path, "a") as f:
                f.write(json.dumps(trace) + "\n")
    except Exception as e:
        log.warning("Failed to write retrieval trace: %s", e)


# Default mode routing (overridden by config.yaml retrieval.modes)
_DEFAULT_MODE_ROUTING = {
    "background": {
        "n_retrieve": 60,
        "allowed_types": {"background", "method"},
        "preferred_sections": {"abstract", "introduction", "background", "related work"},
        "authority_boost": 1.5,
        "review_boost": 0.3,
        "max_per_paper": 2,
    },
    "draft": {
        "n_retrieve": 60,
        "authority_boost": 1.0,
        "review_boost": 0.0,
        "max_per_paper": 3,
    },
    "check": {
        "n_retrieve": 60,
        "preferred_sections": {"results", "experiments", "methods", "discussion"},
        "authority_boost": 1.2,
        "review_boost": 0.0,
        "max_per_paper": 4,
    },
    "review": {
        "n_retrieve": 60,
        "preferred_sections": {"results", "experiments", "methods", "discussion"},
        "authority_boost": 1.2,
        "review_boost": 0.0,
        "max_per_paper": 4,
    },
    "compare": {
        "n_retrieve": 60,
        "preferred_sections": {"results", "experiments", "discussion", "abstract"},
        "authority_boost": 1.0,
        "review_boost": 0.0,
        "max_per_paper": 3,
    },
    "gaps": {
        "n_retrieve": 60,
        "preferred_sections": {"discussion", "conclusion", "limitations", "future work"},
        "authority_boost": 0.8,
        "review_boost": 0.2,
        "max_per_paper": 2,
    },
    "outline": {
        "n_retrieve": 60,
        "authority_boost": 1.0,
        "review_boost": 0.1,
        "max_per_paper": 2,
    },
}
MODE_ROUTING = dict(_DEFAULT_MODE_ROUTING)

# HyDE prompt templates keyed by section category
_HYDE_PROMPTS = {
    "background": (
        "You are an academic research assistant. Given a query about "
        "event cameras, RGB-Event fusion, spiking neural networks, or "
        "autonomous driving, write a textbook-style explanation as if from "
        "a survey paper's background or related work section. Write 2-4 "
        "sentences of clear, foundational academic content. Do not hedge or disclaim."
    ),
    "results": (
        "You are an academic research assistant. Given a query about "
        "event cameras, RGB-Event fusion, spiking neural networks, or "
        "autonomous driving, write a findings paragraph as if from a results "
        "or experiments section. Include specific metrics, dataset names, and "
        "quantitative comparisons. Write 2-4 sentences. Do not hedge or disclaim."
    ),
    "limitations": (
        "You are an academic research assistant. Given a query about "
        "event cameras, RGB-Event fusion, spiking neural networks, or "
        "autonomous driving, write about limitations, open challenges, and "
        "future work directions as if from a discussion or conclusion section. "
        "Write 2-4 sentences. Do not hedge or disclaim."
    ),
    "generic": (
        "You are an academic research assistant. Given a query about "
        "event cameras, RGB-Event fusion, spiking neural networks, or "
        "autonomous driving, write a short hypothetical answer as if it "
        "were extracted from a research paper. Write 2-4 sentences of "
        "factual, specific academic content. Do not hedge or disclaim."
    ),
}

# Section-to-category mapping for HyDE prompt selection
_SECTION_CATEGORIES = {
    "abstract": "background", "introduction": "background",
    "background": "background", "related work": "background",
    "results": "results", "experiments": "results", "methods": "results",
    "discussion": "limitations", "conclusion": "limitations",
    "limitations": "limitations", "future work": "limitations",
}


def _select_hyde_prompt(preferred_sections: set | None) -> str:
    """Select a HyDE prompt template based on preferred sections.

    Picks the category with the most section matches (plurality).
    Tie-breaks to 'generic'.
    """
    if not preferred_sections:
        return _HYDE_PROMPTS["generic"]
    counts: dict[str, int] = {}
    for sec in preferred_sections:
        cat = _SECTION_CATEGORIES.get(sec)
        if cat:
            counts[cat] = counts.get(cat, 0) + 1
    if not counts:
        return _HYDE_PROMPTS["generic"]
    best = max(counts, key=counts.get)
    # Tie-break: if multiple categories have same count, use generic
    max_count = counts[best]
    if sum(1 for c in counts.values() if c == max_count) > 1:
        return _HYDE_PROMPTS["generic"]
    return _HYDE_PROMPTS[best]


def _load_retrieval_config(cfg: dict) -> dict:
    """Load retrieval parameters from config, return the retrieval section."""
    ret = cfg.get("retrieval", {})
    # Merge mode configs from YAML over defaults
    yaml_modes = ret.get("modes", {})
    for mode_name, mode_cfg in yaml_modes.items():
        base = dict(_DEFAULT_MODE_ROUTING.get(mode_name, {}))
        # Convert lists to sets for preferred_sections and allowed_types
        if "preferred_sections" in mode_cfg and isinstance(mode_cfg["preferred_sections"], list):
            mode_cfg["preferred_sections"] = set(mode_cfg["preferred_sections"])
        if "allowed_types" in mode_cfg and isinstance(mode_cfg["allowed_types"], list):
            mode_cfg["allowed_types"] = set(mode_cfg["allowed_types"])
        base.update(mode_cfg)
        MODE_ROUTING[mode_name] = base
    return ret

# Query embedding cache — mutable wrapper so maxsize can be set at init
_embed_cache_size = 256

def _cached_embed(text: str) -> list[float]:
    """Embed a query string with LRU caching.

    Text is wrapped in Q:/A: format with the query instruction prefix.
    For pre-formatted text (e.g. HyDE passages already in Q:/A: format),
    use _cached_embed_raw instead to avoid double-wrapping.
    """
    return _cached_embed_inner(text)

@lru_cache(maxsize=256)
def _cached_embed_inner(text: str) -> list[float]:
    formatted = format_nugget_text(
        {"question": text, "answer": ""}, _embed_instruction
    )
    return _embed_text(formatted)

def _cached_embed_raw(text: str) -> list[float]:
    """Embed pre-formatted text (already in Q:/A: format) with instruction prefix only."""
    return _cached_embed_raw_inner(text)

@lru_cache(maxsize=256)
def _cached_embed_raw_inner(text: str) -> list[float]:
    formatted = f"Instruct: {_embed_instruction}\nQuery: {text}" if _embed_instruction else text
    return _embed_text(formatted)

def _embed_text(formatted: str) -> list[float]:
    kwargs = {"model": _embed_model, "input": [formatted]}
    emb_cfg = _cfg.get("embed", {}).get("embedding", {})
    dims = emb_cfg.get("dimensions")
    if dims:
        kwargs["dimensions"] = dims
    resp = _embed_client.embeddings.create(**kwargs)
    return resp.data[0].embedding

COMMANDS_DIR = Path(__file__).resolve().parent.parent / ".claude" / "commands"
WEB_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
WEB_MODE_FILES = {"background": "background_web.md"}
MODE_FILES = {
    "survey": "survey.md",
    "gaps": "gaps.md",
    "compare": "compare.md",
    "check": "check.md",
    "cite": "cite.md",
    "review": "review.md",
    "draft": "draft.md",
    "outline": "outline.md",
    "background": "background.md",
}


import re as _re

# Patterns for query-type routing (compiled once)
_COMPARE_PATTERNS = _re.compile(
    r'\b(compar|vs\.?|versus|differ|trade.?offs?|advantage|disadvantage|better|worse|'
    r'outperform|benchmark|against|relative to|between .+ and)\b', _re.IGNORECASE)
_DEFINITIONAL_PATTERNS = _re.compile(
    r'\b(what is|what are|define|definition|explain|concept|principle|'
    r'how does .* work|overview|introduction|fundamentals)\b', _re.IGNORECASE)

# KB-derived entity set, populated at startup from nugget corpus
_kb_entities: set[str] = set()

# Common words/abbreviations that match entity patterns but aren't model/dataset names
_ENTITY_STOPWORDS = frozenset({
    # Common English that appear uppercase
    'THE', 'AND', 'FOR', 'WITH', 'FROM', 'THIS', 'THAT', 'ARE', 'NOT',
    'IS', 'IT', 'OR', 'AN', 'AS', 'AT', 'BY', 'IF', 'IN', 'NO', 'OF',
    'ON', 'SO', 'TO', 'UP', 'WE', 'DO', 'BE', 'HE', 'ME',
    # Venues/journals
    'IEEE', 'CVPR', 'ICCV', 'ECCV', 'AAAI', 'NIPS', 'ICML', 'ACCV',
    # Metrics/units (useful for retrieval but not entity-specific)
    'MSE', 'MAE', 'RMSE', 'EPE', 'AP', 'FPS', 'AUC', 'ACC',
    # Hardware/generic tech
    'GPU', 'CPU', 'CMOS', 'DRAM', 'RAM', 'USB', 'PCB',
    # Misc
    'NOS', 'JSO', 'APR2021', 'AB', 'AA', 'AD', 'ADS', 'EV',
})


def _build_entity_set(db_path: str, min_count: int = 10):
    """Build entity set from KB nuggets (acronyms + CamelCase terms).

    Runs once at startup. Extracts terms that look like proper names:
    - Pure acronyms: 2+ uppercase letters (SNN, DSEC, LIF)
    - CamelCase: mixed-case with internal capitals (FlowNet, SpikingJelly)
    - Hyphenated names: ST-FlowNet, Spike-FlowNet, EV-FlowNet

    Filters out common English words and venue/metric names.
    """
    import sqlite3
    from collections import Counter
    conn = sqlite3.connect(db_path)
    cursor = conn.execute('SELECT question, answer FROM nuggets')

    # CamelCase: FlowNet, SpikingJelly (lower after first cap, then internal cap)
    camel_re = _re.compile(r'\b([A-Z][a-z]+(?:[A-Z][a-z0-9]+)+(?:[-][A-Za-z0-9]+)*)\b')
    # AcronymWord: MCFNet, ResNet, YOLOv5 (2+ uppercase then lowercase)
    acroword_re = _re.compile(r'\b([A-Z]{2,}[a-z][A-Za-z0-9]*)\b')
    # Hyphenated names: ST-FlowNet, Spike-FlowNet, EV-FlowNet
    hyphen_re = _re.compile(r'\b([A-Z][A-Za-z0-9]*[-][A-Z][A-Za-z0-9]+(?:[-][A-Za-z0-9]+)*)\b')
    # Pure acronyms: 3+ uppercase (SNN, LIF, DSEC)
    acronym_re = _re.compile(r'\b([A-Z]{3,}[0-9]*(?:[-][A-Z0-9]+)*)\b')

    counts: Counter = Counter()
    for q, a in cursor:
        for text in (q, a):
            for m in camel_re.findall(text):
                counts[m] += 1
            for m in acroword_re.findall(text):
                counts[m] += 1
            for m in hyphen_re.findall(text):
                counts[m] += 1
            for m in acronym_re.findall(text):
                counts[m] += 1
    conn.close()

    entities = set()
    for term, count in counts.items():
        if count >= min_count and term not in _ENTITY_STOPWORDS:
            entities.add(term)
            entities.add(term.lower())
    return entities


def _classify_query(query: str) -> dict:
    """Classify query and return a routing delta dict.

    The delta layers on top of MODE_ROUTING to produce an effective config.
    Keys set to None mean "defer to mode config / global default".

    Returns dict with keys:
        bm25_weight (float): BM25 multiplier for RRF fusion [0.5, 2.0]
        hyde_enabled (bool|None): override HyDE on/off, None = defer
        n_retrieve_scale (float|None): multiplier for mode's n_retrieve
        blend_weight (float|None): cross-encoder blend weight override
        section_prefs (set|None): override mode's preferred_sections
        authority_boost_scale (float|None): multiplier for mode's authority_boost
    """
    is_comparison = bool(_COMPARE_PATTERNS.search(query))
    is_definitional = bool(_DEFINITIONAL_PATTERNS.search(query))

    # Count KB entities in the query
    entity_hits = sum(1 for word in _re.findall(r'\b[\w-]+\b', query)
                      if word in _kb_entities or word.upper() in _kb_entities)

    # --- BM25 weight (same logic as before) ---
    bm25_score = 0.0
    if is_comparison:
        bm25_score += 0.5
    if entity_hits >= 2:
        bm25_score += 0.5
    elif entity_hits == 1:
        bm25_score += 0.25
    if is_definitional:
        bm25_score -= 0.3
    bm25_weight = max(0.5, min(2.0, 1.0 + bm25_score))

    # --- HyDE gating ---
    query_tokens = len(query.split())
    hyde_enabled = None  # defer to global config by default
    hyde_reason = "deferred to config"

    if is_comparison:
        hyde_enabled = True
        hyde_reason = "comparison query (bridging vocabulary)"
    elif entity_hits >= 2:
        hyde_enabled = False
        hyde_reason = f"{entity_hits} KB entities, no comparison"
    elif query_tokens < 8 and entity_hits >= 1:
        hyde_enabled = False
        hyde_reason = f"short query ({query_tokens} tokens) with entity"
    elif is_definitional and entity_hits == 0:
        hyde_enabled = True
        hyde_reason = "definitional query, no entities"

    log.debug("Query classification: bm25=%.2f hyde=%s (%s) entities=%d query=%s",
              bm25_weight, hyde_enabled, hyde_reason, entity_hits, query[:60])

    return {
        "bm25_weight": bm25_weight,
        "hyde_enabled": hyde_enabled,
        "hyde_reason": hyde_reason,
        "entity_hits": entity_hits,
        "is_comparison": is_comparison,
        "n_retrieve_scale": None,
        "blend_weight": None,
        "section_prefs": None,
        "authority_boost_scale": None,
    }


def _build_effective_config(mode: str, query_delta: dict, n_retrieve_default: int = 60) -> dict:
    """Layer query classification delta on top of mode config.

    Precedence: query_delta > mode config > global defaults.
    Multiplicative fields (n_retrieve_scale, authority_boost_scale) multiply
    the mode's absolute value. Clamped to safe ranges.
    """
    base = dict(MODE_ROUTING.get(mode, {}))

    # Direct overrides
    base["bm25_weight"] = query_delta["bm25_weight"]

    if query_delta.get("hyde_enabled") is not None:
        base["hyde_enabled"] = query_delta["hyde_enabled"]
    else:
        base.setdefault("hyde_enabled", _retrieval_cfg.get("hyde_enabled", True))

    base["hyde_reason"] = query_delta.get("hyde_reason", "")

    if query_delta.get("blend_weight") is not None:
        base["blend_weight"] = query_delta["blend_weight"]

    if query_delta.get("section_prefs") is not None:
        base["preferred_sections"] = query_delta["section_prefs"]

    # Multiplicative scales
    if query_delta.get("n_retrieve_scale") is not None:
        scale = max(0.3, min(3.0, query_delta["n_retrieve_scale"]))
        base_n = base.get("n_retrieve", n_retrieve_default)
        base["n_retrieve"] = round(base_n * scale)

    if query_delta.get("authority_boost_scale") is not None:
        scale = max(0.3, min(3.0, query_delta["authority_boost_scale"]))
        base["authority_boost"] = base.get("authority_boost", 1.0) * scale

    return base


def _parse_bib_file(bib_path: str) -> dict[str, str]:
    """Parse a .bib file and return lookup dicts mapping DOI/arXiv/title → bibtex key."""
    lookup: dict[str, str] = {}
    try:
        text = Path(bib_path).expanduser().read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return lookup

    # Split into entries
    entries = re.findall(r"@\w+\{([^,]+),([^@]*)", text, re.DOTALL)
    for cite_key, body in entries:
        cite_key = cite_key.strip()

        # Extract DOI
        doi_m = re.search(r"doi\s*=\s*\{([^}]+)\}", body, re.IGNORECASE)
        if doi_m:
            doi = doi_m.group(1).strip().lower()
            lookup[f"doi:{doi}"] = cite_key

        # Extract arXiv ID from eprint, note, or url fields
        for field in ("eprint", "note", "url"):
            field_m = re.search(rf"{field}\s*=\s*\{{([^}}]+)\}}", body, re.IGNORECASE)
            if field_m:
                arxiv_m = re.search(r"(\d{4}\.\d{4,5})", field_m.group(1))
                if arxiv_m:
                    lookup[f"arxiv:{arxiv_m.group(1)}"] = cite_key
                    break

        # Extract title for fuzzy matching
        title_m = re.search(r"title\s*=\s*\{([^}]+)\}", body, re.IGNORECASE)
        if title_m:
            title = re.sub(r"\s+", " ", title_m.group(1).strip().lower())
            lookup[f"title:{title}"] = cite_key

    return lookup


def _resolve_bibtex_key(arxiv_id: str, doi: str, title: str) -> str | None:
    """Look up a real bibtex key by arXiv ID, DOI, or normalised title."""
    if arxiv_id:
        key = _bib_lookup.get(f"arxiv:{arxiv_id}")
        if key:
            return key
    if doi:
        key = _bib_lookup.get(f"doi:{doi.lower()}")
        if key:
            return key
    if title:
        normalised = re.sub(r"\s+", " ", title.strip().lower())
        key = _bib_lookup.get(f"title:{normalised}")
        if key:
            return key
    return None


def _extract_cite_keys(text: str) -> list[str]:
    r"""Extract individual \cite{} keys from text, splitting comma-separated keys."""
    raw = re.findall(r"\\cite\{([^}]+)\}", text)
    keys = []
    for group in raw:
        for k in group.split(","):
            k = k.strip()
            if k:
                keys.append(k)
    return list(dict.fromkeys(keys))  # deduplicate, preserve order


# arXiv ID: YYMM.NNNNN with plausible year prefix (10xx-24xx)
_ARXIV_RE = re.compile(r'\b((?:1[0-9]|2[0-4])\d{2}\.\d{4,5})\b')
# paper_id: YYMM_NNNNN (underscore variant)
_PAPER_ID_RE = re.compile(r'\b(\d{4}_\d{4,5})\b')


def _extract_paper_refs(text: str) -> list[tuple[str, str]]:
    r"""Extract paper references from text in three forms.

    Returns list of (ref_string, ref_type) tuples where ref_type is
    'cite', 'arxiv', or 'paper_id'.
    """
    refs = []
    seen = set()
    # \cite{} keys first
    for key in _extract_cite_keys(text):
        if key not in seen:
            refs.append((key, "cite"))
            seen.add(key)
    # Bare arXiv IDs
    for m in _ARXIV_RE.finditer(text):
        aid = m.group(1)
        if aid not in seen:
            refs.append((aid, "arxiv"))
            seen.add(aid)
    # paper_id patterns
    for m in _PAPER_ID_RE.finditer(text):
        pid = m.group(1)
        # Convert to arXiv form to dedup with arXiv matches
        arxiv_form = pid.replace("_", ".")
        if pid not in seen and arxiv_form not in seen:
            refs.append((pid, "paper_id"))
            seen.add(pid)
    return refs


def _resolve_paper_ref(ref: str, ref_type: str) -> str | None:
    """Resolve a single paper reference to a paper_id.

    Returns paper_id string or None if not found.
    """
    if not _kb:
        return None
    if ref_type == "cite":
        return _resolve_cite_to_paper_id(ref)
    elif ref_type == "arxiv":
        # Try direct paper_id lookup (arXiv 2401.17151 -> paper_id 2401_17151)
        pid = ref.replace(".", "_")
        row = _kb.db.execute(
            "SELECT paper_id FROM papers WHERE paper_id = ?", (pid,)
        ).fetchone()
        if row:
            return row[0]
        # Try arxiv_id field
        row = _kb.db.execute(
            "SELECT paper_id FROM papers WHERE arxiv_id = ?", (ref,)
        ).fetchone()
        return row[0] if row else None
    elif ref_type == "paper_id":
        row = _kb.db.execute(
            "SELECT paper_id FROM papers WHERE paper_id = ?", (ref,)
        ).fetchone()
        return row[0] if row else None
    return None


def _short_circuit_retrieve(paper_id: str, mode: str) -> list[dict]:
    """Fetch and enrich nuggets directly from SQLite for a single paper.

    Returns enriched nugget dicts matching _run_retrieval output format.
    """
    routing = MODE_ROUTING.get(mode, {})
    allowed_types = routing.get("allowed_types")
    preferred_sections = routing.get("preferred_sections")
    max_per_paper = routing.get("max_per_paper", 20)

    # Fetch nuggets from SQLite
    rows = _kb.get_paper_nuggets(paper_id)
    if not rows:
        return []

    # Filter by allowed_types (hard filter, matching normal pipeline)
    if allowed_types:
        rows = [r for r in rows if r.get("type", "") in allowed_types]

    # Sort: preferred section match first, then thesis_relevance, then overall_score
    def _sort_key(r):
        sec_match = 1 if preferred_sections and r.get("section", "") in preferred_sections else 0
        rel = r.get("thesis_relevance", 0) or 0
        score = r.get("overall_score") if r.get("overall_score") is not None else -1
        return (sec_match, rel, score)

    rows.sort(key=_sort_key, reverse=True)
    rows = rows[:max_per_paper]

    # Get paper metadata
    paper = _kb._get_paper(paper_id)
    if not paper:
        return []

    p_title = paper.get("title", "")
    p_year = paper.get("year")
    p_authors = paper.get("authors", "[]")
    p_arxiv = paper.get("arxiv_id", "")
    p_doi = paper.get("doi", "")

    # Resolve bibtex key
    real_key = _resolve_bibtex_key(p_arxiv, p_doi, p_title)

    nuggets = []
    for r in rows:
        nid = r["nugget_id"]
        doc = f"Q: {r.get('question', '')}\nA: {r.get('answer', '')}"

        # Generate bibtex key if no real one found
        if real_key:
            bib_key = real_key
            bib_status = "real"
        else:
            # Simple fallback key generation for short-circuit path
            if p_arxiv:
                bib_key = f"arXiv_{p_arxiv.replace('.', '_')}"
            elif p_authors and p_authors not in ("[]", ""):
                surname = p_authors.split(",")[0].strip().split()[-1] if p_authors.split(",")[0].strip().split() else "Unknown"
                surname = "".join(c for c in surname if c.isalpha()) or "Unknown"
                yr = str(p_year) if p_year else "XXXX"
                bib_key = f"{surname}_{yr}"
            else:
                bib_key = f"paper_{paper_id}"
            bib_status = "generated"

        nuggets.append({
            "nugget_id": nid,
            "paper_id": paper_id,
            "paper_title": p_title,
            "paper_year": p_year,
            "paper_authors": p_authors,
            "arxiv_id": p_arxiv,
            "doi": p_doi,
            "type": r.get("type", ""),
            "confidence": r.get("confidence", ""),
            "section": r.get("section", ""),
            "document": doc,
            "distance": 0.0,
            "thesis_relevance": r.get("thesis_relevance", 0),
            "rrf_score": 1.0,
            "overlap_count": 0,
            "matched_queries": [],
            "bibtex_key": bib_key,
            "bib_status": bib_status,
            "pinned": False,
            "source_chunk": r.get("source_chunk"),
            "retrieval_mode": "short-circuit",
        })
    return nuggets


def _resolve_cite_to_paper_id(key: str) -> str | None:
    """Resolve a cite key to a paper_id using the bib lookup and SQLite."""
    # Check if this key is a value in _bib_lookup (i.e., it's a real bib key)
    for lookup_key, bib_key in _bib_lookup.items():
        if bib_key == key:
            # Extract the identifier to search papers table
            if lookup_key.startswith("arxiv:"):
                arxiv_id = lookup_key[6:]
                row = _kb.db.execute(
                    "SELECT paper_id FROM papers WHERE arxiv_id = ?", (arxiv_id,)
                ).fetchone()
                if row:
                    return row[0]
            elif lookup_key.startswith("doi:"):
                doi = lookup_key[4:]
                row = _kb.db.execute(
                    "SELECT paper_id FROM papers WHERE LOWER(doi) = ?", (doi,)
                ).fetchone()
                if row:
                    return row[0]
            elif lookup_key.startswith("title:"):
                title = lookup_key[6:]
                row = _kb.db.execute(
                    "SELECT paper_id FROM papers WHERE LOWER(title) = ?", (title,)
                ).fetchone()
                if row:
                    return row[0]

    # Fallback: parse author + year from key like "AuthorLast_YEAR:slug"
    base = key.split(":")[0] if ":" in key else key
    parts = base.split("_")
    author = parts[0] if parts else None
    year = None
    for p in parts:
        if p.isdigit() and len(p) == 4:
            year = int(p)
            break
    if author:
        query = "SELECT paper_id, title FROM papers WHERE authors LIKE ?"
        params: list = [f"%{author}%"]
        if year:
            query += " AND year = ?"
            params.append(year)
        query += " LIMIT 10"
        rows = _kb.db.execute(query, params).fetchall()
        if rows:
            # If there's a slug, score by title match
            if ":" in key:
                slug = key.split(":", 1)[1].lower().replace("-", " ").replace("_", " ")
                slug_words = set(slug.split())
                best_pid, best_score = None, 0
                for r in rows:
                    title_words = set(re.split(r"\W+", (r[1] or "").lower())) - {"", "a", "an", "the", "of", "for", "and", "in", "on", "with", "to", "from", "by"}
                    if slug_words and title_words:
                        overlap = len(slug_words & title_words)
                        score = overlap / len(slug_words)
                        if score > best_score:
                            best_score = score
                            best_pid = r[0]
                if best_pid and best_score > 0.3:
                    return best_pid
            return rows[0][0]
    return None


def _init_feedback_db(kb_dir: str):
    """Initialize feedback SQLite database."""
    global _feedback_db
    import sqlite3
    fb_path = os.path.join(kb_dir, "feedback.db")
    _feedback_db = sqlite3.connect(fb_path, check_same_thread=False)
    _feedback_db.execute("PRAGMA journal_mode=WAL")
    _feedback_db.row_factory = sqlite3.Row
    _feedback_db.execute("""
        CREATE TABLE IF NOT EXISTS nugget_feedback (
            nugget_id TEXT,
            paper_id TEXT,
            rating INTEGER DEFAULT 0,
            query TEXT,
            mode TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (nugget_id, query)
        )
    """)
    _feedback_db.execute("""
        CREATE TABLE IF NOT EXISTS paper_feedback (
            paper_id TEXT PRIMARY KEY,
            rating INTEGER DEFAULT 0,
            updated_at TEXT DEFAULT (datetime('now'))
        )
    """)
    _feedback_db.commit()
    log.info("Feedback DB initialized at %s", fb_path)


def _init(config_path: str):
    global _cfg, _kb, _embed_client, _embed_model, _embed_instruction, _collection, _bib_lookup, _executor, _retrieval_cfg
    log.info("Initializing with config=%s", config_path)
    _cfg = load_config(config_path)
    _retrieval_cfg = _load_retrieval_config(_cfg)
    max_workers = _retrieval_cfg.get("max_workers", 8)
    _executor = ThreadPoolExecutor(max_workers=max_workers)
    log.info("ThreadPoolExecutor: max_workers=%d", max_workers)
    _kb = ThesisKB(config_path)
    _embed_client, _embed_model = make_embed_client(_cfg)
    emb_cfg = _cfg.get("embed", {}).get("embedding", {})
    # Use query_instruction at query time if available
    _embed_instruction = emb_cfg.get("query_instruction", emb_cfg.get("instruction", ""))
    _collection = _kb.collection
    bib_path = _cfg.get("paths", {}).get("bib_file", "")
    _bib_lookup = _parse_bib_file(bib_path) if bib_path else {}
    _init_feedback_db(_cfg["paths"]["kb_dir"])
    global _kb_entities
    sqlite_cfg = _cfg.get("store", {}).get("sqlite", {})
    db_name = sqlite_cfg.get("db_name", "nuggets.db")
    db_path = os.path.join(_cfg["paths"]["kb_dir"], db_name)
    _kb_entities = _build_entity_set(db_path)
    stats = _kb.stats()
    log.info("KB loaded: %d papers, %d nuggets, %d bib keys, %d entities",
             stats["total_papers"], stats["total_nuggets"], len(_bib_lookup),
             len(_kb_entities) // 2)  # //2 because both cases stored


def _shutdown():
    global _kb, _feedback_db
    if _feedback_db:
        _feedback_db.close()
        _feedback_db = None
    if _kb:
        _kb.close()
        _kb = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    _init(_config_path)
    yield
    _shutdown()


app = FastAPI(title="Thesis KB API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    n: int = 10

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    mode: str = "survey"
    n_variants: int = 6
    n_retrieve: int = 60
    n_context: int = 16
    model: str = "minimax/minimax-m2.5"
    latex_mode: bool = False
    year_min: int | None = None
    year_max: int | None = None
    excluded_nuggets: list[str] = []
    excluded_papers: list[str] = []
    type_filter: list[str] = []
    pinned_papers: list[str] = []
    max_per_paper: int = 3
    rerank: bool = True
    rerank_top_n: int = 150
    rerank_weight: float = 0.6

    @field_validator("rerank_top_n")
    @classmethod
    def clamp_rerank_top_n(cls, v):
        return max(1, min(v, 300))

    @field_validator("rerank_weight")
    @classmethod
    def clamp_rerank_weight(cls, v):
        return max(0.0, min(v, 1.0))

class RetrieveRequest(BaseModel):
    query: str
    mode: str = "background"
    n_context: int = 16
    n_variants: int = 6
    n_retrieve: int = 60
    model: str = "minimax/minimax-m2.5"
    year_min: int | None = None
    year_max: int | None = None
    type_filter: list[str] = []
    max_per_paper: int = 3
    rerank: bool = True
    rerank_top_n: int = 150
    rerank_weight: float = 0.6

    @field_validator("rerank_top_n")
    @classmethod
    def clamp_rerank_top_n(cls, v):
        return max(1, min(v, 300))

    @field_validator("rerank_weight")
    @classmethod
    def clamp_rerank_weight(cls, v):
        return max(0.0, min(v, 1.0))

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    try:
        stats = _kb.stats()
        ecfg = _cfg.get("embed", {})
        backend = ecfg.get("backend", "vllm")
        if backend == "ollama":
            emb_model = ecfg.get("ollama", {}).get("model", "")
        else:
            emb_model = ecfg.get("vllm", {}).get("model", "")
        ncfg = _cfg.get("nuggets", {})
        llm_backend = ncfg.get("backend", "vllm")
        if llm_backend == "ollama":
            llm_model = ncfg.get("ollama", {}).get("model", "")
        else:
            llm_model = ncfg.get("vllm", {}).get("model", "")
        return {
            "status": "ok",
            "nuggets": stats["total_nuggets"],
            "papers": stats["total_papers"],
            "embed_backend": backend,
            "embed_model": emb_model,
            "llm_model": llm_model,
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e),
            "nuggets": 0,
            "papers": 0,
            "embed_backend": "",
            "embed_model": "",
            "llm_model": "",
        }


@app.get("/stats")
async def stats_endpoint():
    if not _kb:
        raise HTTPException(503, "KB not initialized")
    try:
        return _kb.stats()
    except Exception as e:
        raise HTTPException(500, str(e))


class FeedbackRequest(BaseModel):
    nugget_id: str
    paper_id: str
    rating: int  # +1 (helpful) or -1 (irrelevant)
    query: str = ""
    mode: str = ""


@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest):
    """Record user feedback on a retrieved nugget."""
    if not _feedback_db:
        raise HTTPException(503, "Feedback DB not initialized")
    with _feedback_lock:
        _feedback_db.execute(
            """INSERT OR REPLACE INTO nugget_feedback (nugget_id, paper_id, rating, query, mode)
               VALUES (?, ?, ?, ?, ?)""",
            (req.nugget_id, req.paper_id, req.rating, req.query, req.mode),
        )
        # Aggregate paper-level rating
        _feedback_db.execute(
            """INSERT INTO paper_feedback (paper_id, rating, updated_at)
               VALUES (?, ?, datetime('now'))
               ON CONFLICT(paper_id) DO UPDATE SET
                 rating = (SELECT SUM(rating) FROM nugget_feedback WHERE paper_id = ?),
                 updated_at = datetime('now')""",
            (req.paper_id, req.rating, req.paper_id),
        )
        _feedback_db.commit()
    return {"status": "ok"}


@app.get("/api/feedback/papers")
async def get_paper_feedback():
    """Get aggregated paper feedback for boosting."""
    if not _feedback_db:
        return {}
    rows = _feedback_db.execute(
        "SELECT paper_id, rating FROM paper_feedback WHERE rating != 0"
    ).fetchall()
    return {r["paper_id"]: r["rating"] for r in rows}


@app.get("/api/feedback/nuggets")
async def get_nugget_feedback():
    """Get nugget-level feedback."""
    if not _feedback_db:
        return {}
    rows = _feedback_db.execute(
        "SELECT nugget_id, SUM(rating) as total FROM nugget_feedback GROUP BY nugget_id HAVING total != 0"
    ).fetchall()
    return {r["nugget_id"]: r["total"] for r in rows}


@app.get("/api/chunk/{paper_id}/{chunk_id}")
async def get_chunk(paper_id: str, chunk_id: int):
    """Load a parent chunk's full text from disk."""
    if not _kb:
        raise HTTPException(503, "KB not initialized")
    text = _kb.load_chunk(paper_id, chunk_id)
    if text is None:
        raise HTTPException(404, "Chunk not found")
    return {"paper_id": paper_id, "chunk_id": chunk_id, "text": text}


@app.get("/api/papers/search")
async def paper_search(q: str = "", limit: int = 10):
    """Search papers by title, author, or year. Each token must match at least one field."""
    if not _kb or not _kb.db:
        raise HTTPException(503, "KB not initialized")
    q = q.strip()
    if not q:
        return []
    tokens = q.split()
    conditions = []
    params = []
    for token in tokens:
        conditions.append(
            "(title LIKE ? OR authors LIKE ? OR CAST(year AS TEXT) LIKE ?)"
        )
        params.extend([f"%{token}%", f"%{token}%", f"%{token}%"])
    where = " AND ".join(conditions)
    rows = _kb.db.execute(
        f"SELECT paper_id, title, authors, year, arxiv_id FROM papers WHERE {where} LIMIT ?",
        params + [limit],
    ).fetchall()
    return [dict(r) for r in rows]


@app.post("/query")
async def query_endpoint(req: QueryRequest):
    if not _kb:
        raise HTTPException(503, "KB not initialized")
    try:
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            _executor, lambda: _kb.query(req.query, n_results=req.n)
        )
        return {"results": results}
    except Exception as e:
        raise HTTPException(500, str(e))


async def _run_retrieval(
    query: str,
    mode: str,
    n_variants: int,
    n_retrieve: int,
    n_context: int,
    model: str,
    type_filter: list[str],
    max_per_paper: int,
    year_min: int | None,
    year_max: int | None,
    rerank: bool,
    rerank_top_n: int,
    rerank_weight: float,
    excluded_nuggets: list[str] | None = None,
    excluded_papers: list[str] | None = None,
    pinned_papers: list[str] | None = None,
) -> tuple[list[dict], list[str]]:
    """Run the full retrieval pipeline and return (enriched_nuggets, variants).

    Shared by /retrieve and /chat endpoints.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(500, "OPENROUTER_API_KEY not set")

    t0 = time.time()
    loop = asyncio.get_running_loop()
    last_msg = query

    # --- Step 0: Classify query and build effective routing config ---
    query_delta = _classify_query(last_msg)
    effective = _build_effective_config(mode, query_delta, n_retrieve_default=n_retrieve)
    log.debug("Effective routing config: mode=%s %s", mode,
              {k: v for k, v in effective.items() if k not in ("hyde_reason",)})

    expand_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        timeout=httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0),
    )

    def _expand():
        # Mode-specific expansion prompts
        if mode == "draft":
            system_content = (
                f"You are a search query expander for an academic knowledge base about "
                f"event-based vision, spiking neural networks, and autonomous driving.\n\n"
                f"Given a topic to draft a thesis paragraph about, generate {n_variants} search queries. "
                f"Focus on finding concrete evidence for writing:\n"
                f"- 3 queries targeting \"method\" nuggets (architectures, algorithms, how things work)\n"
                f"- 2 queries targeting \"result\" nuggets (quantitative performance, benchmarks)\n"
                f"- 1 query targeting \"background\" nuggets (definitions, context)\n"
                f'{"- Fill extra slots with method or result types" if n_variants > 6 else ""}\n\n'
                f'Output ONLY a JSON array of objects: [{{"query": "...", "target_type": "method"}}, ...]\n'
                f"No explanation."
            )
        elif mode == "outline":
            system_content = (
                f"You are a search query expander for an academic knowledge base about "
                f"event-based vision, spiking neural networks, and autonomous driving.\n\n"
                f"Given a thesis section to outline, generate {n_variants} broad search queries "
                f"spanning the full scope of the section. Cast a wide net:\n"
                f"- 1 query targeting \"background\" nuggets (foundational concepts)\n"
                f"- 2 queries targeting \"method\" nuggets (different approaches/techniques)\n"
                f"- 1 query targeting \"result\" nuggets (key findings)\n"
                f"- 1 query targeting \"comparison\" nuggets (trade-offs, alternatives)\n"
                f"- 1 query targeting \"limitation\" nuggets (open problems, gaps)\n"
                f'{"- Fill extra slots with any type" if n_variants > 6 else ""}\n\n'
                f'Output ONLY a JSON array of objects: [{{"query": "...", "target_type": "method"}}, ...]\n'
                f"No explanation."
            )
        elif mode in ("review", "check"):
            system_content = (
                f"You are a claim extractor for an academic knowledge base about "
                f"event-based vision, spiking neural networks, and autonomous driving.\n\n"
                f"Given a piece of thesis text, extract the {n_variants} most important "
                f"factual claims and convert each into a search query to verify it against "
                f"the knowledge base. Focus on:\n"
                f"- Technical claims about how sensors/methods work (target_type: \"method\")\n"
                f"- Quantitative claims with numbers or performance figures (target_type: \"result\")\n"
                f"- Comparative claims (X is better/worse than Y) (target_type: \"comparison\")\n"
                f"- Claims about limitations or challenges (target_type: \"limitation\")\n"
                f"- Definitional or background claims (target_type: \"background\")\n\n"
                f'Output ONLY a JSON array of objects: [{{"query": "...", "target_type": "method"}}, ...]\n'
                f"Each query should be a direct search for evidence supporting or contradicting the claim.\n"
                f"No explanation."
            )
        elif mode == "background":
            system_content = (
                f"You are a search query expander for an academic knowledge base about "
                f"event-based vision, spiking neural networks, and autonomous driving.\n\n"
                f"Given a topic for a neutral background paragraph, generate {n_variants} search queries "
                f"that find FACTUAL, DEFINITIONAL content only. Focus on:\n"
                f"- How the technology/concept works (mechanisms, principles, architecture)\n"
                f"- Measurable properties (specifications, parameters, characteristics)\n"
                f"- Definitions and foundational concepts\n\n"
                f"DO NOT generate queries about:\n"
                f"- Comparisons between technologies\n"
                f"- Limitations, weaknesses, or drawbacks\n"
                f"- Advantages or motivations for alternative approaches\n\n"
                f"Use this distribution:\n"
                f"- {max(n_variants // 2, 1)} queries targeting \"background\" nuggets (definitions, context, foundational concepts)\n"
                f"- {n_variants - max(n_variants // 2, 1)} queries targeting \"method\" nuggets (mechanisms, how things work)\n\n"
                f'Output ONLY a JSON array of objects: [{{"query": "...", "target_type": "background"}}, ...]\n'
                f"No explanation."
            )
        else:
            system_content = (
                f"You are a search query expander for an academic knowledge base about "
                f"event-based vision, spiking neural networks, and autonomous driving.\n\n"
                f"Given a user question, generate {n_variants} search queries. "
                f"Each query MUST target a specific nugget type. Use this distribution:\n"
                f"- 2 queries targeting \"method\" nuggets (how things work, architectures, algorithms)\n"
                f"- 1 query targeting \"result\" nuggets (quantitative performance, benchmarks, metrics)\n"
                f"- 1 query targeting \"comparison\" nuggets (X vs Y, trade-offs, advantages/disadvantages)\n"
                f"- 1 query targeting \"limitation\" nuggets (weaknesses, open problems, failure cases)\n"
                f"- 1 query targeting \"background\" nuggets (definitions, context, foundational concepts)\n"
                f'{"- Fill extra slots with any type" if n_variants > 6 else ""}\n\n'
                f'Output ONLY a JSON array of objects: [{{"query": "...", "target_type": "method"}}, ...]\n'
                f"No explanation."
            )

        resp = expand_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": last_msg},
            ],
            temperature=0.3,
            max_tokens=500,
        )
        text = resp.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [line for line in lines if not line.startswith("```")]
            text = "\n".join(lines).strip()
        return json.loads(text)

    try:
        raw = await loop.run_in_executor(_executor, _expand)
        if not isinstance(raw, list):
            raw = [last_msg]
        if raw and isinstance(raw[0], dict):
            variants = [item.get("query", str(item)) for item in raw]
            variant_types = [item.get("target_type") for item in raw]
        else:
            variants = [str(item) for item in raw]
            variant_types = [None] * len(variants)
    except Exception:
        if mode == "background":
            variants = [
                last_msg,
                f"definition and principles of {last_msg}",
                f"how {last_msg} works mechanism",
                f"properties and characteristics of {last_msg}",
            ]
            variant_types = ["background", "background", "method", "method"]
        else:
            variants = [
                last_msg,
                f"methods and architectures for {last_msg}",
                f"experimental results and benchmarks for {last_msg}",
                f"limitations and challenges of {last_msg}",
            ]
            variant_types = [None, "method", "result", "limitation"]

    log.debug("Expanded to %d variants: %s", len(variants), variants)

    # --- Step 1b: HyDE — generate a hypothetical nugget passage ---
    hyde_enabled = effective.get("hyde_enabled", True)
    log.debug("HyDE %s: %s", "enabled" if hyde_enabled else "disabled",
              effective.get("hyde_reason", ""))

    if hyde_enabled:
        # Select mode-aware HyDE prompt based on effective preferred_sections
        hyde_system_prompt = _select_hyde_prompt(effective.get("preferred_sections"))

        def _generate_hyde():
            try:
                resp = expand_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": hyde_system_prompt},
                        {"role": "user", "content": last_msg},
                    ],
                    temperature=0.4,
                    max_tokens=200,
                )
                answer = resp.choices[0].message.content.strip()
                return f"Q: {last_msg}\nA: {answer}"
            except Exception as e:
                log.warning("HyDE generation failed: %s", e)
                return None

        hyde_task = loop.run_in_executor(_executor, _generate_hyde)

    # --- Step 2: Embed each variant (with LRU cache) ---
    def _embed_one(text):
        return _cached_embed(text)

    t_embed = time.time()
    embed_tasks = [
        loop.run_in_executor(_executor, _embed_one, v) for v in variants
    ]

    # Await HyDE result and embed it (pre-formatted, skip Q:/A: wrapping)
    hyde_passage = None
    if hyde_enabled:
        hyde_passage = await hyde_task
        if hyde_passage:
            variants.append(hyde_passage)
            variant_types.append(None)  # no type filter for HyDE
            embed_tasks.append(loop.run_in_executor(
                _executor, _cached_embed_raw, hyde_passage))
            log.debug("HyDE passage: %s", hyde_passage[:120])

    embeddings = await asyncio.gather(*embed_tasks)
    log.info("Embedded %d variants in %.0fms", len(variants), (time.time() - t_embed) * 1000)

    # --- Step 3: Multi-vector ChromaDB retrieval + BM25 ---
    routing = effective  # unified config replaces raw MODE_ROUTING
    effective_n_retrieve = effective.get("n_retrieve", n_retrieve)

    def _retrieve(vec, target_type=None):
        kwargs = {"query_embeddings": [vec], "n_results": effective_n_retrieve}
        if type_filter:
            if len(type_filter) == 1:
                kwargs["where"] = {"type": type_filter[0]}
            else:
                kwargs["where"] = {"type": {"$in": type_filter}}
        # target_type is no longer used as a hard filter —
        # type preference is applied as an RRF boost instead
        return _collection.query(**kwargs)

    retrieve_tasks = [
        loop.run_in_executor(_executor, _retrieve, emb, vtype)
        for emb, vtype in zip(embeddings, variant_types)
    ]

    # BM25 retrieval (parallel with vector retrieval)
    # Exclude HyDE passage from BM25 — its value is in the vector space, not keywords
    bm25_variants = variants[:-1] if hyde_passage else variants

    def _bm25_search():
        bm25_results = []
        for v in bm25_variants:
            bm25_results.extend(_kb.bm25_search(v, n_results=effective_n_retrieve))
        return bm25_results

    t_retrieve = time.time()
    bm25_task = loop.run_in_executor(_executor, _bm25_search)
    all_results = await asyncio.gather(*retrieve_tasks)
    bm25_raw = await bm25_task
    log.info("Vector+BM25 retrieval in %.0fms", (time.time() - t_retrieve) * 1000)

    # --- Step 4: Reciprocal Rank Fusion ---
    rrf_k = _retrieval_cfg.get("rrf_k", 30)
    rrf_scores: dict[str, float] = {}
    overlap_count: dict[str, int] = {}
    matched_queries: dict[str, list[int]] = {}
    nugget_data: dict[str, dict] = {}

    hyde_weight = _retrieval_cfg.get("hyde_weight", 1.0)
    hyde_qi = len(variants) - 1 if hyde_passage else -1  # index of HyDE variant

    for qi, result_set in enumerate(all_results):
        ids = result_set["ids"][0]
        metas = result_set["metadatas"][0]
        docs = result_set["documents"][0]
        dists = result_set["distances"][0]
        rrf_w = hyde_weight if qi == hyde_qi else 1.0
        for rank, (nid, meta, doc, dist) in enumerate(
            zip(ids, metas, docs, dists)
        ):
            rrf_scores[nid] = rrf_scores.get(nid, 0) + rrf_w / (rank + rrf_k)
            overlap_count[nid] = overlap_count.get(nid, 0) + 1
            matched_queries.setdefault(nid, []).append(qi)
            if nid not in nugget_data:
                nugget_data[nid] = {
                    "nugget_id": nid,
                    "type": meta.get("type", ""),
                    "confidence": meta.get("confidence", ""),
                    "section": meta.get("section", ""),
                    "document": doc,
                    "paper_id": meta.get("paper_id", ""),
                    "distance": round(dist, 4),
                    "thesis_relevance": meta.get("thesis_relevance", 3),
                }

    # Query-type routing: BM25 weight from effective config (set by _classify_query)
    bm25_weight = effective["bm25_weight"]

    # Merge BM25 results into RRF
    bm25_seen = {}
    for nid, score in bm25_raw:
        if nid not in bm25_seen or score > bm25_seen[nid]:
            bm25_seen[nid] = score
    bm25_ranked = sorted(bm25_seen.keys(), key=lambda x: bm25_seen[x], reverse=True)
    for rank, nid in enumerate(bm25_ranked):
        rrf_scores[nid] = rrf_scores.get(nid, 0) + bm25_weight / (rank + rrf_k)
        overlap_count[nid] = overlap_count.get(nid, 0) + 1
        matched_queries.setdefault(nid, []).append(len(variants))
        if nid not in nugget_data:
            row = _kb.db.execute(
                "SELECT * FROM nuggets WHERE nugget_id = ?", (nid,)
            ).fetchone()
            if row:
                r = dict(row)
                nugget_data[nid] = {
                    "nugget_id": nid,
                    "type": r.get("type", ""),
                    "confidence": r.get("confidence", ""),
                    "section": r.get("section", ""),
                    "document": f"Q: {r.get('question', '')}\nA: {r.get('answer', '')}",
                    "paper_id": r.get("paper_id", ""),
                    "distance": 0.0,
                    "thesis_relevance": r.get("thesis_relevance", 3),
                    "source_chunk": r.get("source_chunk"),
                    "overall_score": r.get("overall_score"),
                }
    log.debug("BM25 contributed %d unique nuggets (%d new)",
              len(bm25_ranked), len([n for n in bm25_ranked if n not in nugget_data]))

    # Thesis relevance boosting (0 means unscored — treat as neutral 3)
    for nid in rrf_scores:
        relevance = nugget_data[nid].get("thesis_relevance", 3)
        try:
            relevance = int(relevance)
        except (ValueError, TypeError):
            relevance = 3
        if relevance <= 0:
            relevance = 3  # unscored nuggets get neutral treatment
        rrf_scores[nid] *= 1.0 + (relevance - 3) * 0.2

    # Section-aware boosting (mode-specific)
    section_boost = _retrieval_cfg.get("section_boost", 1.15)
    preferred_sections = routing.get("preferred_sections")
    if preferred_sections:
        for nid in rrf_scores:
            section = nugget_data[nid].get("section", "").lower()
            if any(ps in section for ps in preferred_sections):
                rrf_scores[nid] *= section_boost

    # Type-match boosting: reward nuggets whose type matches a variant's target_type
    type_match_boost = _retrieval_cfg.get("type_match_boost", 1.2)
    if type_match_boost > 1.0:
        # Build set of target types requested by the expanded variants
        target_type_set = {t for t in variant_types if t}
        if target_type_set:
            for nid in rrf_scores:
                if nugget_data[nid].get("type", "") in target_type_set:
                    rrf_scores[nid] *= type_match_boost

    # Feedback boosting (from user's prior ratings)
    if _feedback_db:
        try:
            with _feedback_lock:
                paper_fb = {r[0]: r[1] for r in _feedback_db.execute(
                    "SELECT paper_id, rating FROM paper_feedback WHERE rating != 0"
                ).fetchall()}
                nugget_fb = {r[0]: r[1] for r in _feedback_db.execute(
                    "SELECT nugget_id, SUM(rating) FROM nugget_feedback GROUP BY nugget_id HAVING SUM(rating) != 0"
                ).fetchall()}
            fb_pos = _retrieval_cfg.get("feedback_positive", 1.3)
            fb_neg = _retrieval_cfg.get("feedback_negative", 0.5)
            fb_step = _retrieval_cfg.get("paper_feedback_step", 0.05)
            fb_max = _retrieval_cfg.get("paper_feedback_max", 0.3)
            for nid in rrf_scores:
                nfb = nugget_fb.get(nid, 0)
                if nfb > 0:
                    rrf_scores[nid] *= fb_pos
                elif nfb < 0:
                    rrf_scores[nid] *= fb_neg
                pid = nugget_data[nid]["paper_id"]
                pfb = paper_fb.get(pid, 0)
                if pfb > 0:
                    rrf_scores[nid] *= 1.0 + min(pfb * fb_step, fb_max)
                elif pfb < 0:
                    rrf_scores[nid] *= max(1.0 + pfb * 0.1, 0.3)
        except Exception as e:
            log.warning("Feedback boosting failed: %s", e)

    # Depth-of-coverage: boost papers where many nuggets matched
    paper_hits: dict[str, set[str]] = defaultdict(set)
    for nid in rrf_scores:
        paper_hits[nugget_data[nid]["paper_id"]].add(nid)

    depth_frac_w = _retrieval_cfg.get("depth_fraction_weight", 0.3)
    depth_div = _retrieval_cfg.get("depth_abs_divisor", 5)
    depth_max = _retrieval_cfg.get("depth_abs_max", 0.3)
    paper_depth: dict[str, float] = {}
    for pid, nids in paper_hits.items():
        total = _kb.paper_nugget_count(pid) or 1
        raw_hits = len(nids)
        hit_fraction = min(raw_hits / total, 1.0)
        abs_boost = min(math.log2(max(raw_hits, 1)) / depth_div, depth_max)
        paper_depth[pid] = 1.0 + hit_fraction * depth_frac_w + abs_boost

    for nid in rrf_scores:
        pid = nugget_data[nid]["paper_id"]
        rrf_scores[nid] *= paper_depth.get(pid, 1.0)

    # Paper authority boost (also penalizes papers with missing metadata)
    auth_scale = routing.get("authority_boost", 1.0)
    review_boost_val = routing.get("review_boost", 0.0)
    _authority_cache: dict[str, float] = {}
    for nid in rrf_scores:
        pid = nugget_data[nid]["paper_id"]
        if pid not in _authority_cache:
            paper = _kb._get_paper(pid)
            if paper:
                # Penalize papers with broken metadata (no year, placeholder title)
                has_year = paper.get("year") is not None
                title = paper.get("title", "")
                has_real_title = bool(title) and not re.match(r"^\d{4}[\s_]\d{4,5}$", title)
                if not has_year or not has_real_title:
                    _authority_cache[pid] = 0.3  # heavy penalty
                    continue
                cc = paper.get("citation_count") or 0
                icc = paper.get("influential_citation_count") or 0
                ptype = (paper.get("paper_type") or "").lower()
                is_review = "review" in ptype or "survey" in ptype
                authority = 1.0 + auth_scale * 0.1 * min(math.log10(max(cc, 1)), 4)
                if icc > 10:
                    authority += 0.1
                if is_review:
                    authority += review_boost_val
                _authority_cache[pid] = authority
            else:
                _authority_cache[pid] = 1.0
        rrf_scores[nid] *= _authority_cache[pid]

    # Title-match boost: when query mentions a paper/dataset name, boost that paper
    title_boost_val = _retrieval_cfg.get("title_match_boost", 2.0)
    if title_boost_val > 1.0:
        # Extract significant query words (3+ chars, skip common stopwords)
        _title_stops = {"the", "and", "for", "with", "from", "that", "this", "are",
                        "how", "what", "does", "which", "where", "when", "who", "why",
                        "has", "have", "been", "its", "not", "can", "was", "were",
                        "based", "using", "between", "about", "into", "over", "than"}
        query_words = set()
        for w in re.findall(r'\b[A-Za-z][\w-]*\b', last_msg):
            if len(w) >= 3 and w.lower() not in _title_stops:
                query_words.add(w.lower())

        # Build paper_id -> title mapping from authority cache lookup (already loaded)
        title_boosted = set()
        for pid in set(nugget_data[nid]["paper_id"] for nid in rrf_scores):
            paper = _kb._get_paper(pid)
            if not paper:
                continue
            title = (paper.get("title") or "").lower()
            if not title:
                continue
            # Check for significant word overlap between query and title
            title_words = set(w for w in re.findall(r'\b\w+\b', title)
                              if len(w) >= 3 and w not in _title_stops)
            matches = query_words & title_words
            # Require at least one distinctive match (not just common technical words)
            if len(matches) >= 1 and any(len(m) >= 4 for m in matches):
                title_boosted.add(pid)

        if title_boosted:
            for nid in rrf_scores:
                if nugget_data[nid]["paper_id"] in title_boosted:
                    rrf_scores[nid] *= title_boost_val
            log.debug("Title-match boost (%.1fx) applied to %d papers: %s",
                       title_boost_val, len(title_boosted),
                       [p[:40] for p in list(title_boosted)[:5]])

    # Quality score boost — use overall_score from SQLite
    quality_weight = _retrieval_cfg.get("quality_weight", 0.15)
    if quality_weight > 0 and _kb and _kb.db:
        nids_needing_quality = [nid for nid in rrf_scores if "overall_score" not in nugget_data.get(nid, {})]
        if nids_needing_quality:
            # Chunk to stay within SQLite's variable limit (default 999)
            for i in range(0, len(nids_needing_quality), 900):
                chunk = nids_needing_quality[i:i + 900]
                placeholders = ",".join("?" * len(chunk))
                rows = _kb.db.execute(
                    f"SELECT nugget_id, overall_score FROM nuggets WHERE nugget_id IN ({placeholders})",
                    chunk,
                ).fetchall()
                for row in rows:
                    nid_r = row[0] if isinstance(row, (tuple, list)) else row["nugget_id"]
                    score_r = row[1] if isinstance(row, (tuple, list)) else row["overall_score"]
                    if nid_r in nugget_data:
                        nugget_data[nid_r]["overall_score"] = score_r
        for nid in rrf_scores:
            qs = nugget_data.get(nid, {}).get("overall_score")
            if qs is not None and qs > 0:
                # Boost relative to median quality (3.0): score 5 -> 1.3x, score 1 -> 0.7x
                quality_boost = 1.0 + (qs - 3.0) * quality_weight
                rrf_scores[nid] *= max(0.5, quality_boost)

    # Cross-encoder reranking
    if rerank:
        rerank_timeout = _retrieval_cfg.get("rerank_timeout", 10)
        effective_blend = effective.get("blend_weight", rerank_weight)
        pre_rerank = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        rrf_scores = await loop.run_in_executor(
            _executor,
            lambda: rerank_nuggets(
                query=last_msg,
                nugget_ids=pre_rerank,
                nugget_data=nugget_data,
                rrf_scores=rrf_scores,
                top_n=rerank_top_n,
                blend_weight=effective_blend,
                timeout=rerank_timeout,
            ),
        )
        log.info("Reranked top %d candidates (weight=%.1f)", rerank_top_n, effective_blend)

    # Sort by RRF score
    ranked = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    # Exclude rejected nuggets and papers
    _excluded_nuggets = set(excluded_nuggets or [])
    _excluded_papers = set(excluded_papers or [])
    if _excluded_nuggets or _excluded_papers:
        ranked = [
            nid for nid in ranked
            if nid not in _excluded_nuggets
            and nugget_data[nid]["paper_id"] not in _excluded_papers
        ]

    # Mode-specific type restrictions
    allowed_types = routing.get("allowed_types")
    if allowed_types:
        ranked = [nid for nid in ranked if nugget_data[nid]["type"] in allowed_types]

    # Type-balanced + paper-diverse selection
    by_type = defaultdict(list)
    for nid in ranked:
        ntype = nugget_data[nid]["type"]
        by_type[ntype].append(nid)

    selected = []
    paper_counts: dict[str, int] = defaultdict(int)
    max_pp = routing.get("max_per_paper", max_per_paper)

    def _can_add(nid):
        return paper_counts[nugget_data[nid]["paper_id"]] < max_pp

    sorted_types = sorted(by_type.keys(), key=lambda t: rrf_scores[by_type[t][0]], reverse=True)
    for t in sorted_types:
        added = 0
        for nid in by_type[t]:
            if added >= 2 or len(selected) >= n_context:
                break
            if nid not in selected and _can_add(nid):
                selected.append(nid)
                paper_counts[nugget_data[nid]["paper_id"]] += 1
                added += 1
        if len(selected) >= n_context:
            break

    # Backfill respecting max_per_paper
    for nid in ranked:
        if len(selected) >= n_context:
            break
        if nid not in selected and _can_add(nid):
            selected.append(nid)
            paper_counts[nugget_data[nid]["paper_id"]] += 1

    # Final backfill ignoring max_per_paper only if still short
    if len(selected) < n_context:
        for nid in ranked:
            if len(selected) >= n_context:
                break
            if nid not in selected:
                selected.append(nid)
                paper_counts[nugget_data[nid]["paper_id"]] += 1

    top_ids = selected

    # Inject pinned paper nuggets
    pinned_nids = []
    if pinned_papers:
        for pid in pinned_papers:
            rows = _kb.get_paper_nuggets(pid)
            if not rows:
                continue
            rows.sort(key=lambda r: r.get("thesis_relevance", 3), reverse=True)
            type_counts_pin = defaultdict(int)
            paper_nuggets = []
            for r in rows:
                ntype = r.get("type", "")
                if type_counts_pin[ntype] >= 4:
                    continue
                type_counts_pin[ntype] += 1
                paper_nuggets.append(r)
                if len(paper_nuggets) >= 15:
                    break
            for r in paper_nuggets:
                nid = r["nugget_id"]
                if nid in nugget_data or nid in pinned_nids:
                    continue
                nugget_data[nid] = {
                    "nugget_id": nid,
                    "type": r.get("type", ""),
                    "confidence": r.get("confidence", ""),
                    "section": r.get("section", ""),
                    "document": f"Q: {r.get('question', '')}\nA: {r.get('answer', '')}",
                    "paper_id": pid,
                    "distance": 0.0,
                    "thesis_relevance": r.get("thesis_relevance", 3),
                    "pinned": True,
                }
                rrf_scores[nid] = 1.0
                overlap_count[nid] = 0
                matched_queries[nid] = []
                pinned_nids.append(nid)
        if len(pinned_nids) > 20:
            pinned_nids.sort(key=lambda n: nugget_data[n].get("thesis_relevance", 3), reverse=True)
            pinned_nids = pinned_nids[:20]
        top_ids = [nid for nid in top_ids if nid not in pinned_nids]
        max_rrf = max(4, n_context - len(pinned_nids))
        top_ids = pinned_nids + top_ids[:max_rrf]
        log.info("Pinned %d nuggets from %d papers", len(pinned_nids), len(pinned_papers))

    log.info("Retrieved %d unique nuggets, selected %d after RRF+balancing",
             len(nugget_data), len(top_ids))

    # Enrich with paper metadata from SQLite
    bibtex_key_counts: dict[str, int] = {}
    paper_id_to_bibtex_key: dict[str, str] = {}  # cache: same paper → same key

    def _make_bibtex_key(authors: str, year, arxiv_id: str = "", title: str = "", paper_id: str = "") -> str:
        # Return cached key if same paper_id already processed
        if paper_id and paper_id in paper_id_to_bibtex_key:
            return paper_id_to_bibtex_key[paper_id]
        # Extract surname
        surname = ""
        if authors and authors not in ("[]", ""):
            first = authors.split(",")[0].strip()
            parts = first.split()
            if parts:
                surname = parts[-1]
                surname = "".join(c for c in surname if c.isalpha())
        if not surname and arxiv_id:
            # Fallback: use arXiv ID as key base
            key = f"arXiv_{arxiv_id.replace('.', '_')}"
            if paper_id:
                paper_id_to_bibtex_key[paper_id] = key
            return key
        if not surname:
            surname = "Unknown"
        yr = str(year) if year else "XXXX"
        # Build slug from title
        slug = ""
        if title and title != title.replace(" ", "_"):  # not just a paper_id
            words = re.split(r"\W+", title.lower())
            stop = {"a", "an", "the", "of", "for", "and", "in", "on", "with", "to", "from", "by", "is", "are", "was", "were"}
            slug_words = [w for w in words if w and w not in stop][:4]
            if slug_words:
                slug = "-".join(slug_words)
        base = f"{surname}_{yr}"
        if slug:
            base = f"{base}:{slug}"
        count = bibtex_key_counts.get(base, 0)
        bibtex_key_counts[base] = count + 1
        key = base if count == 0 else f"{base}-{chr(ord('a') + count - 1)}"
        if paper_id:
            paper_id_to_bibtex_key[paper_id] = key
        return key

    def _enrich(nid):
        nd = nugget_data[nid]
        paper = _kb._get_paper(nd["paper_id"])
        nd["paper_title"] = paper.get("title", "") if paper else ""
        nd["paper_year"] = paper.get("year") if paper else None
        nd["paper_authors"] = paper.get("authors", "") if paper else ""
        nd["arxiv_id"] = paper.get("arxiv_id", "") if paper else ""
        nd["doi"] = paper.get("doi", "") if paper else ""
        real_key = _resolve_bibtex_key(nd["arxiv_id"], nd["doi"], nd["paper_title"])
        nd["bibtex_key"] = real_key if real_key else _make_bibtex_key(
            nd["paper_authors"], nd["paper_year"], nd["arxiv_id"], nd["paper_title"],
            paper_id=nd["paper_id"]
        )
        nd["bib_status"] = "real" if real_key else "generated"
        nd["rrf_score"] = round(rrf_scores[nid], 4)
        nd["overlap_count"] = overlap_count[nid]
        nd["matched_queries"] = matched_queries.get(nid, [])
        if "source_chunk" not in nd:
            row = _kb.db.execute(
                "SELECT source_chunk FROM nuggets WHERE nugget_id = ?", (nid,)
            ).fetchone()
            nd["source_chunk"] = row[0] if row else None
        return nd

    top_nuggets = [_enrich(nid) for nid in top_ids]

    # Year filtering (post-enrichment since we need paper_year)
    if year_min or year_max:
        top_nuggets = [
            n for n in top_nuggets
            if n["paper_year"] is not None
            and (year_min is None or n["paper_year"] >= year_min)
            and (year_max is None or n["paper_year"] <= year_max)
        ]

    elapsed = time.time() - t0
    log.info("Retrieval complete: %d nuggets, %.1fs", len(top_nuggets), elapsed)

    # --- Retrieval trace logging ---
    _write_retrieval_trace(
        query=last_msg, mode=mode, elapsed=elapsed,
        variants=variants, effective_config=effective,
        query_delta=query_delta,
        hyde_passage=hyde_passage if hyde_passage else None,
        rrf_scores=rrf_scores, nugget_data=nugget_data,
        overlap_count=overlap_count,
        top_ids=top_ids,
    )

    return top_nuggets, variants


@app.post("/retrieve")
async def retrieve_endpoint(req: RetrieveRequest):
    """Run the full retrieval pipeline and return ranked nuggets as JSON.

    Same pipeline as /chat (query expansion, RRF, reranking, diversity caps)
    but returns sources directly without LLM generation.
    """
    if not _kb:
        raise HTTPException(503, "KB not initialized")
    try:
        top_nuggets, variants = await _run_retrieval(
            query=req.query,
            mode=req.mode,
            n_variants=req.n_variants,
            n_retrieve=req.n_retrieve,
            n_context=req.n_context,
            model=req.model,
            type_filter=req.type_filter,
            max_per_paper=req.max_per_paper,
            year_min=req.year_min,
            year_max=req.year_max,
            rerank=req.rerank,
            rerank_top_n=req.rerank_top_n,
            rerank_weight=req.rerank_weight,
        )
        results = []
        for n in top_nuggets:
            results.append({
                "nugget_id": n["nugget_id"],
                "paper_id": n["paper_id"],
                "paper_title": n["paper_title"],
                "paper_year": n["paper_year"],
                "paper_authors": n["paper_authors"],
                "arxiv_id": n["arxiv_id"],
                "type": n["type"],
                "section": n["section"],
                "document": n["document"],
                "rrf_score": float(n["rrf_score"]),
                "overlap_count": int(n["overlap_count"]),
                "matched_queries": [int(q) for q in n["matched_queries"]],
                "n_variants": len(variants),
                "confidence": n["confidence"],
                "thesis_relevance": int(n.get("thesis_relevance", 3)),
                "bibtex_key": n["bibtex_key"],
                "bib_status": n.get("bib_status", "generated"),
                "pinned": bool(n.get("pinned", False)),
                "distance": float(n.get("distance", 0.0)),
            })
        return {"results": results, "variants": variants}
    except HTTPException:
        raise
    except Exception as e:
        log.error("Retrieve error: %s", e, exc_info=True)
        raise HTTPException(500, str(e))


@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    if not _kb:
        raise HTTPException(503, "KB not initialized")

    try:
        last_msg = req.messages[-1].content if req.messages else ""
        log.info("Chat request: mode=%s model=%s query=%s",
                 req.mode, req.model, last_msg[:100])

        # --- Paper-reference short-circuit ---
        # Detect paper references and short-circuit for single-paper queries
        paper_refs = _extract_paper_refs(last_msg)
        resolved_papers = []
        if paper_refs:
            for ref, rtype in paper_refs:
                pid = _resolve_paper_ref(ref, rtype)
                if pid and pid not in [p for p, _ in resolved_papers]:
                    resolved_papers.append((pid, ref))

            has_cross_paper = bool(_COMPARE_PATTERNS.search(last_msg))

            if (len(resolved_papers) == 1
                    and not has_cross_paper
                    and len(paper_refs) == len(resolved_papers)):
                # Single paper, all refs resolved, no cross-paper intent -> short-circuit
                sc_pid = resolved_papers[0][0]
                sc_nuggets = _short_circuit_retrieve(sc_pid, req.mode)
                if sc_nuggets:
                    log.info("Short-circuit: paper_id=%s, %d nuggets returned",
                             sc_pid, len(sc_nuggets))
                    top_nuggets, variants = sc_nuggets, []
                else:
                    # Fallback: paper exists but no nuggets, pin and do full retrieval
                    log.info("Short-circuit fallback: paper_id=%s has no matching nuggets",
                             sc_pid)
                    if sc_pid not in req.pinned_papers:
                        req.pinned_papers.append(sc_pid)
                    top_nuggets = None  # signal to run full retrieval below
            else:
                # Multiple papers or cross-paper intent: pin and do full retrieval
                top_nuggets = None
                for pid, ref in resolved_papers:
                    if pid not in req.pinned_papers:
                        req.pinned_papers.append(pid)
                        log.info("Paper ref '%s' resolved to paper_id=%s (pinned)", ref, pid)
        else:
            top_nuggets = None

        # --- Full retrieval (when short-circuit didn't fire or fell back) ---
        if top_nuggets is None:
            # Legacy cite-pinning for modes that use it (extends short-circuit pinning)
            if req.mode in ("background", "draft", "review", "check"):
                cite_keys = _extract_cite_keys(last_msg)
                if cite_keys:
                    existing_pinned = set(req.pinned_papers)
                    for ck in cite_keys:
                        pid = _resolve_cite_to_paper_id(ck)
                        if pid and pid not in existing_pinned:
                            req.pinned_papers.append(pid)
                            existing_pinned.add(pid)
                            log.info("Cite key '%s' resolved to paper_id=%s (pinned)", ck, pid)

            top_nuggets, variants = await _run_retrieval(
                query=last_msg,
                mode=req.mode,
                n_variants=req.n_variants,
                n_retrieve=req.n_retrieve,
                n_context=req.n_context,
                model=req.model,
                type_filter=req.type_filter,
                max_per_paper=req.max_per_paper,
                year_min=req.year_min,
                year_max=req.year_max,
                rerank=req.rerank,
                rerank_top_n=req.rerank_top_n,
                rerank_weight=req.rerank_weight,
                excluded_nuggets=req.excluded_nuggets,
                excluded_papers=req.excluded_papers,
                pinned_papers=req.pinned_papers,
            )

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise HTTPException(500, "OPENROUTER_API_KEY not set")

        t0 = time.time()
        loop = asyncio.get_running_loop()
        # --- Step 5: Build context + call OpenRouter ---
        web_file = WEB_MODE_FILES.get(req.mode, "")
        web_path = WEB_PROMPTS_DIR / web_file if web_file else None
        if web_path and web_path.exists():
            mode_prompt = web_path.read_text()
            log.info("Using web prompt: %s", web_file)
        else:
            mode_file = MODE_FILES.get(req.mode, "survey.md")
            mode_path = COMMANDS_DIR / mode_file
            if mode_path.exists():
                mode_prompt = mode_path.read_text()
            else:
                mode_prompt = (COMMANDS_DIR / "survey.md").read_text()

        sources_xml = "<sources>\n"
        for n in top_nuggets:
            pinned_attr = ' pinned="true"' if n.get("pinned") else ""
            bib_status = n.get("bib_status", "generated")
            sources_xml += (
                f'<source bibtex_key="{n["bibtex_key"]}" bib_status="{bib_status}" '
                f'paper="{n["paper_title"]}" year="{n["paper_year"]}" type="{n["type"]}" '
                f'overlap="{n["overlap_count"]}/{len(variants)}"{pinned_attr}>\n'
                f'{n["document"]}\n'
                f"</source>\n"
            )
        sources_xml += "</sources>\n\n"

        system_prompt = sources_xml + mode_prompt

        if req.latex_mode or req.mode == "background":
            key_table = "\n".join(
                f"  \\cite{{{n['bibtex_key']}}} → {n['paper_title']} ({n['paper_year']})"
                for n in top_nuggets
            )
            system_prompt += (
                "\n\n## LaTeX Citation Mode\n"
                "Use \\cite{bibtex_key} for ALL citations instead of (arXiv:XXXX, YEAR).\n"
                "Use the bibtex_key attribute from each <source> tag.\n"
                "Citation key mapping:\n" + key_table + "\n"
            )

        chat_messages = [{"role": "system", "content": system_prompt}]
        for m in req.messages:
            chat_messages.append({"role": m.role, "content": m.content})

        or_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0),
        )

        # --- Step 6: Stream response ---
        log.info("Streaming response: %d context nuggets, %.1fs retrieval",
                 len(top_nuggets), time.time() - t0)

        async def _stream():
            max_retries = 2
            accumulated = ""

            for attempt in range(max_retries):
                def _call_openrouter(msgs=chat_messages, tokens=4000):
                    return or_client.chat.completions.create(
                        model=req.model,
                        messages=msgs,
                        stream=True,
                        max_tokens=tokens,
                        temperature=0.3,
                    )

                # On retry, append partial content so LLM continues from where it stopped
                if attempt > 0 and accumulated:
                    log.info("Stream retry %d, continuing from %d chars", attempt, len(accumulated))
                    retry_messages = chat_messages + [
                        {"role": "assistant", "content": accumulated}
                    ]
                    remaining_tokens = max(1000, 4000 - len(accumulated) // 3)
                    stream = await loop.run_in_executor(
                        _executor, lambda: _call_openrouter(retry_messages, remaining_tokens)
                    )
                else:
                    stream = await loop.run_in_executor(_executor, _call_openrouter)

                try:
                    for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta.content:
                            delta = chunk.choices[0].delta.content
                            accumulated += delta
                            yield f"data: {json.dumps({'type': 'delta', 'content': delta})}\n\n"
                    break  # success — exit retry loop
                except Exception as e:
                    log.warning("Stream error (attempt %d/%d): %s", attempt + 1, max_retries, e)
                    if attempt == max_retries - 1:
                        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
                        return  # stop streaming — don't send sources after error
                    # otherwise retry

            # Send sources
            sources_payload = []
            for n in top_nuggets:
                sources_payload.append({
                    "nugget_id": n["nugget_id"],
                    "paper_id": n["paper_id"],
                    "paper_title": n["paper_title"],
                    "paper_year": n["paper_year"],
                    "paper_authors": n["paper_authors"],
                    "arxiv_id": n["arxiv_id"],
                    "type": n["type"],
                    "section": n["section"],
                    "document": n["document"],
                    "rrf_score": n["rrf_score"],
                    "overlap_count": n["overlap_count"],
                    "matched_queries": n["matched_queries"],
                    "n_variants": len(variants),
                    "confidence": n["confidence"],
                    "thesis_relevance": n.get("thesis_relevance", 3),
                    "bibtex_key": n["bibtex_key"],
                    "bib_status": n.get("bib_status", "generated"),
                    "pinned": n.get("pinned", False),
                    "source_chunk": n.get("source_chunk"),
                })
            yield f"data: {json.dumps({'type': 'sources', 'nuggets': sources_payload, 'variants': variants})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        return StreamingResponse(_stream(), media_type="text/event-stream")

    except HTTPException:
        raise
    except Exception as e:
        log.error("Chat error: %s", e, exc_info=True)
        err_str = str(e).lower()
        if "connection" in err_str or "refused" in err_str:
            raise HTTPException(
                503,
                "Embedding server offline. Start Ollama with: ollama serve",
            )
        raise HTTPException(500, str(e))


def set_config(path: str):
    """Allow startup script to override config path before lifespan runs."""
    global _config_path
    _config_path = path


if __name__ == "__main__":
    import argparse
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("-p", "--port", type=int, default=8001)
    args = ap.parse_args()
    os.environ["THESIS_KB_CONFIG"] = args.config
    _config_path = args.config
    uvicorn.run("src.api:app", host="0.0.0.0", port=args.port, reload=False)
