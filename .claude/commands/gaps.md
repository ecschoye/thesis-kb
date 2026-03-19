Identify research gaps and potential contributions by analyzing limitations in the knowledge base.

## Input
Research area to analyze: $ARGUMENTS

## Instructions

1. **API mode check:** If the local API server is running (check with `curl -s http://127.0.0.1:8001/health`), add `--api --mode survey` to query commands for higher-quality retrieval (query expansion, RRF fusion, reranking, diversity caps).

1. Search for limitations and claims about the topic:
   ```
   cd /cluster/work/ecschoye/thesis-kb && source .venv/bin/activate && python -m src.query "$ARGUMENTS" -n 20 --types limitation --json
   python -m src.query "$ARGUMENTS" -n 15 --types claim --json
   ```

2. Search for methods to understand what HAS been done:
   ```
   python -m src.query "$ARGUMENTS" -n 20 --types method --json
   ```

3. Search for "future work" and "open problem" related content:
   ```
   python -m src.query "$ARGUMENTS future work open problems" -n 15 --json
   python -m src.query "$ARGUMENTS challenges unsolved" -n 15 --json
   ```

4. Cross-reference: for each limitation found, check if other papers in the KB have addressed it.

## Output Format

### Research Gaps: [Topic]

#### Recurring Limitations
For each limitation mentioned by multiple papers:
- **[Limitation theme]**: Mentioned by N papers
  - "[specific limitation text]" — arXiv:XXXX.XXXXX (YEAR)
  - "[specific limitation text]" — arXiv:XXXX.XXXXX (YEAR)
  - Status: **Unaddressed** / **Partially addressed by [paper]** / **Addressed by [paper]**

#### Single-Paper Limitations
[Limitations mentioned by only one paper but potentially significant]

#### Identified Gaps
Based on the analysis above, these are gaps where your thesis could contribute:

1. **[Gap description]**
   - Evidence: [which limitations point to this gap]
   - What exists: [current approaches and their shortcomings]
   - Potential angle: [how this gap could be addressed]

2. ...

#### What's Well-Covered
[Areas where the KB shows strong coverage with multiple approaches — less opportunity for novel contribution]

## Important
- Convert paper_id underscores to arXiv dots
- A gap is strongest when multiple independent papers identify the same limitation
- Do NOT over-interpret — report what the KB evidence says, let the user decide significance
- Focus on gaps relevant to the thesis topic (RGB-Event fusion, SNNs, event cameras, object detection on embedded platforms)
- End your response with 2-3 suggested follow-up questions under a `### Suggested Follow-ups` heading