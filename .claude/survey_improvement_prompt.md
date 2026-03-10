# Task: Implement Survey Mode Improvements

Implement all 7 changes from `.claude/survey_improvement_plan.md`. Read that file first for full context.

## Files to modify

1. **`src/api.py`** — 5 changes
2. **`static/index.html`** — 1 change
3. **`.claude/commands/survey.md`** — 1 change

## Detailed instructions

### src/api.py

**Change 1: ChatRequest defaults (line ~84-90)**
- `n_variants`: 4 → 6
- `n_retrieve`: 12 → 20
- `n_context`: 8 → 16

**Change 2: Query expansion prompt (line ~165-190)**

Replace the `_expand()` function's system prompt. The new prompt must:
- Ask for `{req.n_variants}` queries as JSON array of objects `[{"query": "...", "target_type": "method"}, ...]`
- Specify distribution: 2 method, 1 result, 1 comparison, 1 limitation, 1 background (adjust if n_variants differs)
- After parsing, extract both `variants` (list of query strings) and `variant_types` (list of target_type strings, same order)
- Fallback: if parsing fails or response is a plain string array, set `variant_types = [None] * len(variants)`

**Change 3: Type-filtered retrieval (line ~217-226)**

Modify `_retrieve` to accept an optional `target_type` parameter:
```python
def _retrieve(vec, target_type=None):
    kwargs = {"query_embeddings": [vec], "n_results": req.n_retrieve}
    if target_type:
        kwargs["where"] = {"type": target_type}
    return _collection.query(**kwargs)
```

Update the retrieval task creation to pass the corresponding type:
```python
retrieve_tasks = [
    loop.run_in_executor(_executor, _retrieve, emb, vtype)
    for emb, vtype in zip(embeddings, variant_types)
]
```

**Change 4: Type-balanced selection (line ~254-256)**

Replace `top_ids = ranked[: req.n_context]` with round-robin type-balanced selection:
- Group ranked nuggets by their `type` from `nugget_data`
- First pass: pick up to 2 from each type present (guarantees diversity)
- Second pass: fill remaining slots by pure RRF score
- Result: `top_ids = selected`

**Change 5: Add type to sources XML (line ~280-289)**

Add `type="{n["type"]}"` as an attribute in the `<source>` tag, right after the `year` attribute.

### static/index.html

**Change 6: Frontend updates**

- Line ~180: Change default model from `"google/gemini-flash-1.5"` to `"google/gemini-3-flash-preview"`
- Line ~378-379: Change variants slider max from 6 to 8
- Line ~393-394: Change context nuggets slider max from 16 to 30
- Line ~183: Change default `n_context` from 8 to 16
- Line ~182: Change default `n_variants` from 4 to 6

### .claude/commands/survey.md

**Change 7: Add type-mapping instructions**

Add a new section before the `## Important` section:

```
## Source Types
Sources are tagged with their nugget type (method, result, comparison, limitation, background, claim).
Use these types to organize your review:
- "method" and "background" sources → Methods and Approaches + Overview sections
- "result" and "comparison" sources → Key Results section
- "limitation" sources → Limitations and Open Challenges section
- "claim" sources → wherever contextually relevant
Ensure each section draws primarily from its corresponding source types.
```

## Testing

After all changes, run:
```bash
bash scripts/start_local.sh
```
Then open `static/index.html` and test with: "operating principles of frame-based cameras"

Verify:
- Settings panel shows new defaults (6 variants, 16 context)
- Response includes diverse nugget types in sources
- No crashes on the expand/retrieve/select pipeline
