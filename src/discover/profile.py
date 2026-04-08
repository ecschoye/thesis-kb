"""Dynamic thesis query profile — rebuilt from KB state each run."""

import os
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from src.acquire.zotero import _norm
from src.utils import load_config, load_json

# Terms too generic to be useful as search queries
STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "has", "have", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "not", "no", "nor", "this",
    "that", "these", "those", "it", "its", "we", "our", "they", "their",
    "which", "what", "who", "how", "when", "where", "than", "then", "also",
    "each", "both", "such", "more", "most", "other", "some", "any", "all",
    "between", "through", "during", "before", "after", "above", "below",
    "up", "down", "out", "off", "over", "under", "into", "about", "against",
    "using", "based", "used", "proposed", "show", "shown", "result", "results",
    "method", "methods", "approach", "paper", "work", "study", "propose",
    "achieve", "perform", "demonstrate", "present", "introduce", "novel",
    "new", "first", "two", "three", "one", "however", "et", "al", "fig",
    "table", "section", "figure", "respectively", "compared", "different",
    "experimental", "use", "high", "low", "large", "small", "well", "still",
    "many", "much", "per", "while", "very", "only", "even", "able", "see",
    "given", "without", "within", "across", "further", "previous", "recently",
    "existing", "following", "several", "various", "due", "thus", "hence",
    "overall", "general", "specifically", "particular", "provide", "applied",
    "unlike", "input", "output", "layer", "layers", "module", "model",
    "models", "network", "networks", "feature", "features", "image", "images",
    "data", "dataset", "datasets", "training", "test", "testing", "task",
    "tasks", "performance", "accuracy", "loss", "number", "time", "set",
    # QA-template noise words (from nugget generation prompts)
    "specific", "primary", "main", "key", "describe", "explain", "detail",
    "limitations", "limitation", "motivation", "hyperparameters", "hyperparameter",
    "architectural", "formulation", "mathematical", "technique", "techniques",
    "implementation", "contribution", "contributions", "comparison", "analysis",
    "evaluation", "experiment", "experiments", "challenge", "challenges",
    "advantage", "advantages", "disadvantage", "disadvantages", "role",
})

# Minimum bigram frequency to be considered a topic term
MIN_BIGRAM_FREQ = 15
# Number of top terms to extract
TOP_TERMS = 40
# How many queries to generate
MAX_QUERIES = 15


@dataclass
class ThesisProfile:
    """Dynamic query profile rebuilt from KB + bibliography each run."""
    queries: list[str] = field(default_factory=list)
    seed_arxiv_ids: list[str] = field(default_factory=list)
    seed_s2_ids: list[str] = field(default_factory=list)
    exclude_arxiv_ids: set[str] = field(default_factory=set)
    exclude_dois: set[str] = field(default_factory=set)
    exclude_title_norms: set[str] = field(default_factory=set)
    keyword_terms: list[str] = field(default_factory=list)


# Default path to local paper PDFs (may not yet be in the KB)
LOCAL_PAPERS_DIR = os.path.expanduser("~/Documents/thesis-papers")


def _extract_bigrams(text):
    """Extract lowercase bigrams from text, filtering stopwords."""
    words = re.findall(r"[a-z][a-z\-]+", text.lower())
    words = [w for w in words if w not in STOPWORDS and len(w) > 2]
    bigrams = []
    for i in range(len(words) - 1):
        bigrams.append(f"{words[i]} {words[i+1]}")
    return bigrams


def _extract_unigrams(text):
    """Extract lowercase unigrams, filtering stopwords."""
    words = re.findall(r"[a-z][a-z\-]+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 3]


def _extract_terms_from_kb(db_path, recent_days=30):
    """Extract top topic terms from KB nuggets via bigram frequency.

    Papers added in the last `recent_days` get 3x weight so the profile
    adapts as new papers are ingested.
    """
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row

    bigram_counts = Counter()
    unigram_counts = Counter()

    # Get recent paper IDs for boosting
    recent_ids = set()
    try:
        rows = db.execute(
            "SELECT paper_id FROM papers WHERE date_added >= date('now', ?)",
            (f"-{recent_days} days",),
        ).fetchall()
        recent_ids = {r["paper_id"] for r in rows}
    except Exception:
        pass  # papers table may not have date_added

    # Sample nuggets (questions are more topical than answers)
    rows = db.execute(
        "SELECT paper_id, question FROM nuggets LIMIT 50000"
    ).fetchall()

    for row in rows:
        text = row["question"]
        weight = 3 if row["paper_id"] in recent_ids else 1
        for bg in _extract_bigrams(text):
            bigram_counts[bg] += weight
        for ug in _extract_unigrams(text):
            unigram_counts[ug] += weight

    db.close()

    # Filter bigrams by minimum frequency
    top_bigrams = [
        term for term, count in bigram_counts.most_common(TOP_TERMS * 2)
        if count >= MIN_BIGRAM_FREQ
    ][:TOP_TERMS]

    # Also keep top unigrams that are domain-specific (high frequency)
    top_unigrams = [
        term for term, count in unigram_counts.most_common(100)
        if count >= MIN_BIGRAM_FREQ * 2
    ][:20]

    return top_bigrams, top_unigrams


def _terms_to_queries(bigrams, unigrams):
    """Convert top terms into effective search queries.

    Strategy: use bigrams directly as queries (they're already 2-word phrases),
    and combine unrelated bigrams to form broader queries.
    """
    queries = []
    seen = set()

    # First pass: use each bigram as a standalone query
    for bg in bigrams[:MAX_QUERIES]:
        # Skip if both words are the same root
        words = bg.split()
        if len(set(words)) < 2:
            continue
        if bg not in seen:
            queries.append(bg)
            seen.add(bg)

    return queries[:MAX_QUERIES]


def _parse_bib_arxiv_ids(bib_path):
    """Extract arXiv IDs from a .bib file (eprint, url, note fields)."""
    arxiv_ids = []
    if not os.path.exists(bib_path):
        return arxiv_ids

    with open(bib_path) as f:
        text = f.read()

    # Match eprint = {XXXX.XXXXX} or arXiv:XXXX.XXXXX in url/note
    for m in re.finditer(r"eprint\s*=\s*\{([^}]+)\}", text):
        aid = m.group(1).strip()
        if re.match(r"\d{4}\.\d{4,5}", aid):
            arxiv_ids.append(aid)

    for m in re.finditer(r"arxiv\.org/abs/(\d{4}\.\d{4,5})", text, re.I):
        aid = m.group(1)
        if aid not in arxiv_ids:
            arxiv_ids.append(aid)

    for m in re.finditer(r"arXiv[:\s]+(\d{4}\.\d{4,5})", text):
        aid = m.group(1)
        if aid not in arxiv_ids:
            arxiv_ids.append(aid)

    return arxiv_ids


def _extract_title_from_pdf_name(filename):
    """Extract the title portion from 'Author - Year - Title.pdf' naming."""
    stem = filename
    if stem.lower().endswith(".pdf"):
        stem = stem[:-4]
    # Split on ' - ' and take everything after the second separator (the title)
    parts = stem.split(" - ")
    if len(parts) >= 3:
        return " - ".join(parts[2:]).strip()
    elif len(parts) == 2:
        return parts[1].strip()
    return stem.strip()


def _scan_local_papers(papers_dir):
    """Scan local PDF directory for titles to exclude.

    Papers in ~/Documents/thesis-papers may not yet be processed into the KB,
    but we still don't want to recommend them as "new" discoveries.
    """
    title_norms = set()
    if not os.path.isdir(papers_dir):
        return title_norms
    for fname in os.listdir(papers_dir):
        if not fname.lower().endswith(".pdf"):
            continue
        title = _extract_title_from_pdf_name(fname)
        if title and len(title) > 5:
            title_norms.add(_norm(title))
    return title_norms


def build_profile(config_path="config.yaml", bib_path=None, recent_days=30,
                  local_papers_dir=None):
    """Build a ThesisProfile from current KB state + bibliography + local PDFs.

    Re-extracts terms on every call so the profile adapts as papers are added.
    """
    cfg = load_config(config_path)
    kb_dir = cfg["paths"]["kb_dir"]
    sqlite_cfg = cfg.get("store", {}).get("sqlite", {})
    db_path = os.path.join(kb_dir, sqlite_cfg.get("db_name", "nuggets.db"))
    manifest_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "corpus", "manifest.json",
    )

    if bib_path is None:
        bib_path = os.path.expanduser(
            "~/TDT4900-master-thesis/bibtex/bibliography.bib"
        )
    if local_papers_dir is None:
        local_papers_dir = LOCAL_PAPERS_DIR

    profile = ThesisProfile()

    # 1a. Build exclusion sets from manifest (KB-processed papers)
    if os.path.exists(manifest_path):
        manifest = load_json(manifest_path)
        for p in manifest:
            aid = p.get("arxiv_id")
            if aid:
                profile.exclude_arxiv_ids.add(aid)
            doi = p.get("doi")
            if doi:
                profile.exclude_dois.add(doi.lower())
            title = p.get("title", "")
            if title:
                profile.exclude_title_norms.add(_norm(title))

    # 1b. Also exclude papers that exist locally but aren't yet in KB
    local_titles = _scan_local_papers(local_papers_dir)
    profile.exclude_title_norms.update(local_titles)
    if local_titles:
        print(f"  Scanned {len(local_titles)} local PDFs for dedup")

    # 2. Extract seed arXiv IDs from bibliography
    profile.seed_arxiv_ids = _parse_bib_arxiv_ids(bib_path)

    # Also add highly-cited papers from manifest as seeds
    if os.path.exists(manifest_path):
        manifest = load_json(manifest_path)
        cited = sorted(
            [p for p in manifest if p.get("citation_count")],
            key=lambda p: p["citation_count"],
            reverse=True,
        )
        for p in cited[:30]:
            aid = p.get("arxiv_id")
            if aid and aid not in profile.seed_arxiv_ids:
                profile.seed_arxiv_ids.append(aid)

    # 3. Extract dynamic topic terms from KB
    if os.path.exists(db_path):
        bigrams, unigrams = _extract_terms_from_kb(db_path, recent_days)
        profile.keyword_terms = bigrams + unigrams
        profile.queries = _terms_to_queries(bigrams, unigrams)
    else:
        # Fallback: use core thesis terms
        profile.queries = [
            "spiking neural network object detection",
            "event camera RGB fusion",
            "neuromorphic optical flow",
            "event-based motion compensation",
            "SNN energy efficiency autonomous driving",
        ]
        profile.keyword_terms = [
            "spiking", "neuromorphic", "event camera", "optical flow",
            "object detection", "dvs", "rgb-event", "motion compensation",
        ]

    return profile
