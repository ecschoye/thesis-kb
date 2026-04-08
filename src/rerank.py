"""Cross-encoder reranker using BGE-reranker-v2-m3 via transformers."""
import time
import torch
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.log import get_logger

log = get_logger("rerank", "api.log")

# Singleton model/tokenizer (loads once)
_model = None
_tokenizer = None
_load_failed: bool = False


def get_reranker(model_name: str = "BAAI/bge-reranker-v2-m3"):
    """Get or create the singleton reranker model. Returns (model, tokenizer) or (None, None)."""
    global _model, _tokenizer, _load_failed
    if _load_failed:
        return None, None
    if _model is None:
        log.info("Loading reranker model: %s", model_name)
        try:
            t0 = time.time()
            _tokenizer = AutoTokenizer.from_pretrained(model_name)
            _model = AutoModelForSequenceClassification.from_pretrained(model_name)
            _model.eval()
            log.info("Reranker loaded in %.1fs", time.time() - t0)
        except Exception as e:
            log.error("Failed to load reranker model %s: %s", model_name, e)
            _load_failed = True
            return None, None
    return _model, _tokenizer


def rerank_nuggets(
    query: str,
    nugget_ids: list[str],
    nugget_data: dict[str, dict],
    rrf_scores: dict[str, float],
    top_n: int = 60,
    blend_weight: float = 0.6,
    model_name: str = "BAAI/bge-reranker-v2-m3",
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
    model, tokenizer = get_reranker(model_name)
    if model is None:
        log.warning("Reranker unavailable, returning RRF-only scores")
        return rrf_scores

    # Only rerank top candidates (cross-encoder is O(n) per query)
    candidates = nugget_ids[:top_n]
    if not candidates:
        return rrf_scores

    # Build query-passage pairs
    pairs = []
    for nid in candidates:
        text = nugget_data[nid].get("document", "")
        pairs.append([query, text])

    t0 = time.time()
    try:
        # Use thread pool with timeout (signal.alarm doesn't work outside main thread)
        def _score():
            with torch.no_grad():
                inputs = tokenizer(pairs, padding=True, truncation=True,
                                   return_tensors="pt", max_length=512)
                return model(**inputs, return_dict=True).logits.view(-1).float()

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_score)
            try:
                scores = future.result(timeout=timeout)
            except FuturesTimeoutError:
                future.cancel()
                raise
    except (FuturesTimeoutError, Exception) as e:
        elapsed = time.time() - t0
        log.warning("Reranking failed after %.0fms, falling back to RRF-only: %s",
                     elapsed * 1000, e)
        return rrf_scores

    elapsed = time.time() - t0
    log.debug("Reranked %d candidates in %.0fms", len(candidates), elapsed * 1000)

    # Build cross-encoder score map
    ce_scores = {}
    for i, nid in enumerate(candidates):
        ce_scores[nid] = scores[i].item()

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
