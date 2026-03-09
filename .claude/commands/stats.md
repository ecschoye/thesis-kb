Show knowledge base statistics and coverage analysis.

## Input
Optional topic focus: $ARGUMENTS

## Instructions

### If no topic specified (general stats):
1. Get overall KB statistics:
   ```
   cd /cluster/work/ecschoye/thesis-kb && source .venv/bin/activate && python -m src.query --stats --json
   ```

2. Format the statistics clearly.

### If a topic is specified:
1. Get overall stats first:
   ```
   cd /cluster/work/ecschoye/thesis-kb && source .venv/bin/activate && python -m src.query --stats --json
   ```

2. Search for nuggets on the topic:
   ```
   python -m src.query "$ARGUMENTS" -n 50 --json
   ```

3. Analyze the results:
   - Count unique papers contributing to this topic
   - Break down by nugget type
   - List the top contributing papers (most nuggets returned)
   - Identify which years are most represented

## Output Format

### KB Statistics

| Metric | Count |
|--------|-------|
| Total papers | X |
| Total nuggets | X |

#### Nuggets by Type
| Type | Count | % |
|------|-------|---|
| method | X | X% |
| result | X | X% |
| ... | ... | ... |

#### Papers by Year
| Year | Papers |
|------|--------|
| 2020 | X |
| ... | ... |

### Topic Coverage (if topic specified)

**Topic:** [topic]
**Relevant nuggets found:** X (top 50 by similarity)
**Unique papers:** X

#### Top Contributing Papers
1. "Paper Title" (arXiv:XXXX.XXXXX, YEAR) — X relevant nuggets
2. ...

#### Type Distribution for This Topic
| Type | Count |
|------|-------|
| ... | ... |

## Important
- Convert paper_id underscores to arXiv dots
- For topic-specific stats, count is based on the top 50 most similar nuggets, not exhaustive
- Percentages should be rounded to 1 decimal place