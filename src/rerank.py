"""Cross-encoder reranker for improving retrieval precision."""
import time
import signal
from flashrank import Ranker, RerankRequest
from src.log import get_logger

log = get_logger("rerank", "api.log")

# Singleton ranker instance (model loads once)
_ranker: Ranker | None = None
_ranker_failed: bool = False  # True if model loading failed — skip future attempts


def get_ranker(model_name: str = "ms-marco-MiniLM-L-12-v2") -> Ranker | None:
    """Get or create the singleton Ranker instance. Returns None on load failure."""
    global _ranker, _ranker_failed
    if _ranker_failed:
        return None
    if _ranker is None:
        log.info("Loading reranker model: %s", model_name)
        try:
            t0 = time.time()
            _ranker = Ranker(model_name=model_name)
            log.info("Reranker loaded in %.1fs", time.time() - t0)
        except Exception as e:
            log.error("Failed to load reranker model %s: %s", model_name, e)
            _ranker_failed = True
            return None
    return _ranker


def rerank_nuggets(
    query: str,
    nugget_ids: list[str],
    nugget_data: dict[str, dict],
    rrf_scores: dict[str, float],
    top_n: int = 60,
    blend_weight: float = 0.6,
    model_name: str = "ms-marco-MiniLM-L-12-v2",
    timeout: int = 10,
) -> dict[str, float]:
    """Rerank nuggets using a cross-encoder and blend with RRF scores.

    Args:
        query: The original user query.
        nugget_ids: Ordered list of nugget IDs (RRF-ranked).
        nugget_data: Dict mapping nugget_id -> nugget metadata (must have 'document').
        rrf_scores: Dict mapping nugget_id -> RRF score.
        top_n: Number of top candidates to rerank (rest keep RRF scores).
        blend_weight: Weight for cross-encoder score (0-1). Higher = trust reranker more.
        model_name: Cross-encoder model name.
        timeout: Max seconds for reranking before falling back to RRF-only.

    Returns:
        Updated rrf_scores dict with blended scores.
    """
    ranker = get_ranker(model_name)
    if ranker is None:
        log.warning("Reranker unavailable, returning RRF-only scores")
        return rrf_scores

    # Only rerank top candidates (cross-encoder is O(n) per query)
    candidates = nugget_ids[:top_n]
    if not candidates:
        return rrf_scores

    # Build passages for flashrank
    passages = []
    for nid in candidates:
        text = nugget_data[nid].get("document", "")
        passages.append({"id": nid, "text": text})

    t0 = time.time()
    try:
        request = RerankRequest(query=query, passages=passages)

        # Use signal-based timeout on Unix
        def _timeout_handler(signum, frame):
            raise TimeoutError(f"Reranking exceeded {timeout}s timeout")

        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout)
        try:
            results = ranker.rerank(request)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    except (TimeoutError, Exception) as e:
        elapsed = time.time() - t0
        log.warning("Reranking failed after %.0fms, falling back to RRF-only: %s",
                     elapsed * 1000, e)
        return rrf_scores

    elapsed = time.time() - t0
    log.debug("Reranked %d candidates in %.0fms", len(candidates), elapsed * 1000)

    # Build cross-encoder score map (normalised to 0-1 range)
    ce_scores = {}
    for r in results:
        ce_scores[r["id"]] = r["score"]

    # Normalise cross-encoder scores to match RRF score scale
    if ce_scores:
        max_ce = max(ce_scores.values())
        min_ce = min(ce_scores.values())
        ce_range = max_ce - min_ce if max_ce > min_ce else 1.0
        max_rrf = max(rrf_scores[nid] for nid in candidates) if candidates else 1.0

        for nid in candidates:
            ce_norm = ((ce_scores.get(nid, 0) - min_ce) / ce_range) * max_rrf
            rrf_orig = rrf_scores[nid]
            rrf_scores[nid] = (1 - blend_weight) * rrf_orig + blend_weight * ce_norm

    return rrf_scores
