"""Backfill remaining metadata: LLM title extraction + S2 re-enrichment."""
import json, os, re, sys, time, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.acquire.enrich import enrich_via_arxiv_id, _s2_headers, _title_similarity, _extract_result, S2_API, S2_FIELDS
from openai import OpenAI
import requests

MANIFEST = "corpus/manifest.json"
TEXT_DIR = "corpus/texts"
# Lower threshold — partial titles still match if the key words are there
MIN_SIM = 0.60


def s2_search_relaxed(title, min_sim=MIN_SIM, max_retries=2):
    """S2 title search with a relaxed similarity threshold."""
    if not title or len(title.strip()) < 10:
        return None
    base = f"{S2_API}/paper/search"
    params = {"query": title[:200], "limit": 5, "fields": S2_FIELDS}
    for attempt in range(max_retries):
        try:
            resp = requests.get(base, params=params, headers=_s2_headers(), timeout=15)
            if resp.status_code == 429:
                time.sleep(2 ** (attempt + 1))
                continue
            resp.raise_for_status()
            results = resp.json().get("data", [])
            if not results:
                return None
            # Pick best match above relaxed threshold
            best_p, best_sim = None, 0
            for candidate in results:
                sim = _title_similarity(title, candidate.get("title", ""))
                if sim > best_sim:
                    best_sim = sim
                    best_p = candidate
            if best_p and best_sim >= min_sim:
                return _extract_result(best_p)
            return None
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return None
    return None


def get_first_page(paper_id):
    """Get first page text for a paper."""
    path = os.path.join(TEXT_DIR, f"{paper_id}.json")
    if not os.path.exists(path):
        return ""
    try:
        with open(path) as f:
            data = json.load(f)
        pages = data.get("pages", [])
        if not pages:
            return ""
        return pages[0].get("text", "")
    except Exception:
        return ""


def extract_titles_llm_batch(papers, client, model):
    """Use LLM to extract titles for papers with bad/missing titles."""
    fixed = 0
    for i, paper in enumerate(papers):
        pid = paper["paper_id"]
        text = get_first_page(pid)[:1500]
        if len(text.strip()) < 30:
            continue

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": (
                        "Extract the full paper title from the given academic paper text. "
                        "The title is usually in the first few lines, often in larger font. "
                        "Ignore author names, affiliations, emails, DOIs, journal headers, "
                        "and conference names. Reply with ONLY the complete title, nothing else. "
                        "If you cannot determine the title, reply NONE."
                    )},
                    {"role": "user", "content": text},
                ],
                temperature=0,
                max_tokens=200,
            )
            title = (resp.choices[0].message.content or "").strip().strip('"\'')
            if title and title.upper() != "NONE" and len(title) >= 10:
                paper["_llm_title"] = title
                fixed += 1
        except Exception as e:
            print(f"    LLM error for {pid[:40]}: {e}")
            time.sleep(2)

        if (i + 1) % 50 == 0:
            print(f"  LLM titles: {i+1}/{len(papers)} ({fixed} extracted)")
        time.sleep(0.2)

    print(f"  LLM extracted {fixed}/{len(papers)} titles")
    return fixed


def main():
    with open(MANIFEST) as f:
        manifest = json.load(f)

    missing = [p for p in manifest if not p.get("abstract") or not p.get("year") or not p.get("authors")]
    print(f"Papers still missing metadata: {len(missing)}")

    # Setup LLM client
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set")
        sys.exit(1)
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    llm_model = "xiaomi/mimo-v2-flash"

    # Phase 1: LLM title extraction for ALL missing papers
    print(f"[1/2] Extracting titles via LLM for {len(missing)} papers...")
    llm_fixed = extract_titles_llm_batch(missing, client, llm_model)

    # Save LLM titles checkpoint
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Checkpoint saved")

    # Phase 2: S2 enrichment using LLM titles (with relaxed threshold)
    print(f"[2/2] Re-enriching {len(missing)} papers via S2 (threshold={MIN_SIM})...")
    enriched = 0
    failed = 0
    for i, paper in enumerate(missing):
        # Use LLM title if available, fall back to existing title
        search_title = paper.get("_llm_title") or paper.get("title", "")
        arxiv_id = paper.get("arxiv_id")

        result = None
        # Try title search first (with LLM-extracted title)
        if search_title and len(search_title) >= 10:
            result = s2_search_relaxed(search_title)

        # Fall back to arXiv ID
        if not result and arxiv_id:
            result = enrich_via_arxiv_id(arxiv_id)

        if result:
            if result.get("s2_title"):
                paper["title"] = result["s2_title"]
            if not paper.get("arxiv_id"):
                paper["arxiv_id"] = result.get("arxiv_id")
            if not paper.get("doi"):
                paper["doi"] = result.get("doi")
            if not paper.get("abstract"):
                paper["abstract"] = result.get("abstract", "")
            if not paper.get("authors") or not paper["authors"]:
                paper["authors"] = result.get("authors", [])
                paper["authors_str"] = ", ".join(result.get("authors", []))
            if not paper.get("year"):
                paper["year"] = result.get("year")
            paper["citation_count"] = result.get("citation_count", 0)
            paper["influential_citation_count"] = result.get("influential_citation_count", 0)
            paper["publication_types"] = result.get("publication_types", [])
            enriched += 1
        else:
            failed += 1
            if search_title:
                print(f"    MISS: {search_title[:70]}")

        if (i + 1) % 50 == 0:
            print(f"  S2: {i+1}/{len(missing)} ({enriched} enriched, {failed} failed)")
            with open(MANIFEST, "w") as f:
                json.dump(manifest, f, indent=2)

        time.sleep(0.5)

    # Clean up temp keys and save
    for p in manifest:
        p.pop("_llm_title", None)
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)

    # Stats
    n_title = sum(1 for p in manifest if p.get("title") and p["title"] != p["paper_id"].replace("_", " ") and len(p["title"]) > 10)
    n_abs = sum(1 for p in manifest if (p.get("abstract") or "").strip())
    n_auth = sum(1 for p in manifest if p.get("authors"))
    n_year = sum(1 for p in manifest if p.get("year"))
    still_missing = sum(1 for p in manifest if not p.get("abstract") or not p.get("year") or not p.get("authors"))
    print(f"\nDone: llm_titles={llm_fixed}, s2_enriched={enriched}, s2_failed={failed}")
    print(f"Manifest: title={n_title}, abstract={n_abs}, authors={n_auth}, year={n_year} / {len(manifest)}")
    print(f"Still missing full metadata: {still_missing}")


if __name__ == "__main__":
    main()
