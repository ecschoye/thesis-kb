# Survey Mode Improvement Plan

## Problem

The web UI survey mode returns shallow, poorly-diversified results:
- 4 query variants × 12 retrieved = ~30 unique candidates → only 8 sent to LLM
- No nugget type filtering — results cluster around whatever type best matches the embedding
- Query expansion prompt is generic ("cover different angles") instead of type-aware
- Sources XML doesn't prominently surface nugget type for LLM organization
- Frontend caps sliders at 6 variants / 16 context nuggets — too low for survey mode

Concrete example: querying "operating principles of frame-based cameras" returned 8 nuggets, half about event camera advantages rather than frame-based operation. Only 6 unique papers.

## Changes

### 1. Type-aware query expansion (src/api.py, `_expand()`)

**Current** prompt:
```
Generate {n_variants} distinct search queries that together cover different angles
```

**New** prompt:
```
You are a search query expander for an academic knowledge base about event-based vision,
spiking neural networks, and autonomous driving.

Given a user question, generate {n_variants} search queries. Each query MUST target a
specific nugget type. Use this distribution:
- 2 queries targeting "method" nuggets (how things work, architectures, algorithms)
- 1 query targeting "result" nuggets (quantitative performance, benchmarks, metrics)
- 1 query targeting "comparison" nuggets (X vs Y, trade-offs, advantages/disadvantages)
- 1 query targeting "limitation" nuggets (weaknesses, open problems, failure cases)
- 1 query targeting "background" nuggets (definitions, context, foundational concepts)

Output a JSON array of objects: [{"query": "...", "target_type": "method"}, ...]
```

Parse the response to extract both query strings and target types.

### 2. Type-filtered retrieval (src/api.py, step 3)

**Current**: Each variant embedding queries ChromaDB with no `where` clause.

**New**: For each expanded query that has a `target_type`, add a ChromaDB `where` filter:
```python
def _retrieve(vec, target_type=None):
    kwargs = {"query_embeddings": [vec], "n_results": req.n_retrieve}
    if target_type:
        kwargs["where"] = {"type": target_type}
    return _collection.query(**kwargs)
```

This ensures each retrieval lane returns nuggets of the intended type.

### 3. Increase default parameters (src/api.py, `ChatRequest`)

| Parameter | Old | New | Rationale |
|-----------|-----|-----|-----------|
| `n_variants` | 4 | 6 | Match 6 type-aware lanes |
| `n_retrieve` | 12 | 20 | More candidates per lane |
| `n_context` | 8 | 16 | More nuggets in LLM context |

### 4. Type-balanced selection after RRF (src/api.py, step 4)

After RRF scoring, instead of just taking top-N by score, ensure type diversity:

```python
# Group by type
by_type = defaultdict(list)
for nid in ranked:
    ntype = nugget_data[nid]["type"]
    by_type[ntype].append(nid)

# Round-robin selection: pick top from each type, then fill remaining by score
selected = []
types_present = list(by_type.keys())
# First pass: 1-2 per type (guarantee diversity)
for t in types_present:
    for nid in by_type[t][:2]:
        if nid not in selected:
            selected.append(nid)
        if len(selected) >= req.n_context:
            break

# Second pass: fill remaining slots by pure RRF score
for nid in ranked:
    if len(selected) >= req.n_context:
        break
    if nid not in selected:
        selected.append(nid)

top_ids = selected
```

### 5. Add nugget type to sources XML (src/api.py, step 5)

**Current**:
```xml
<source id="..." paper="..." year="..." overlap="..." score="...">
```

**New**:
```xml
<source id="..." paper="..." year="..." type="method" overlap="..." score="...">
```

Add `type` as a prominent attribute so the LLM can organize output by type.

### 6. Update frontend slider ranges (static/index.html)

| Slider | Old range | New range | New default |
|--------|-----------|-----------|-------------|
| Query variants | 1-6 | 1-8 | 6 |
| Context nuggets | 4-16 | 4-30 | 16 |

Also update the default model in the frontend settings state:
```js
model: "google/gemini-3-flash-preview"  // was gemini-flash-1.5
```

### 7. Update survey prompt (`.claude/commands/survey.md`)

Add instruction to the prompt telling the LLM about the type-aware sources:

```
Sources are tagged with their nugget type (method, result, comparison, limitation, background, claim).
Organize your review using these types:
- Use "method" sources for the Methods and Approaches section
- Use "result" and "comparison" sources for Key Results
- Use "limitation" sources for Limitations and Open Challenges
- Use "background" sources for the Overview
- Use "claim" sources wherever they are relevant
```

## File Changes Summary

| File | Changes |
|------|---------|
| `src/api.py` | Steps 1-5: new expand prompt, type-filtered retrieval, new defaults, type-balanced selection, type in XML |
| `static/index.html` | Step 6: slider ranges, default model |
| `.claude/commands/survey.md` | Step 7: type-aware organization instructions |

## Risks / Considerations

- **Type filtering may reduce recall**: If a query's target type has few matching nuggets, the filtered retrieval could return low-quality results. Mitigation: fall back to unfiltered if fewer than 5 results returned for a type.
- **LLM expansion format change**: The new structured JSON format (`{"query": ..., "target_type": ...}`) needs robust parsing with fallback to the old string-array format if parsing fails.
- **Token budget**: 16 nuggets × ~200 tokens each = ~3200 tokens of context. Well within limits for Gemini Flash.
- **Latency**: 6 variants instead of 4 means 2 more embedding + retrieval calls, but these run in parallel so impact is minimal.
