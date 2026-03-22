Compare methods or approaches across papers using evidence from the thesis knowledge base.

## Input
What to compare: $ARGUMENTS

## Instructions

1. Search for result and comparison nuggets using the `semantic_search` MCP tool:
   `semantic_search(query="$ARGUMENTS", n=30, types=["result", "comparison"])`

2. Also search for method descriptions to understand what's being compared:
   `semantic_search(query="$ARGUMENTS", n=20, types=["method"])`

3. If the user specified metrics (e.g., "accuracy, latency"), run additional targeted queries:
   `semantic_search(query="[topic] accuracy benchmark", n=15, types=["result"])`
   `semantic_search(query="[topic] latency inference speed", n=15, types=["result"])`

4. Extract quantitative data from nuggets: accuracy, mAP, FPS, latency, parameters, FLOPs, energy consumption, etc.

5. Build a comparison table organized by method/paper.

## Output Format

### Comparison: [Topic]

#### Methods Overview
Brief description of each method/approach found, with citation.

#### Comparison Table

| Method | Paper | Year | Dataset | [Metric 1] | [Metric 2] | [Metric 3] |
|--------|-------|------|---------|-------------|-------------|-------------|
| ... | arXiv:... | ... | ... | ... | ... | ... |

#### Key Findings
- [Bullet points summarizing the main takeaways from the comparison]
- Which method performs best on which metric
- Trade-offs between methods (e.g., accuracy vs. speed)

#### Caveats
- [Note where results are not directly comparable: different datasets, evaluation protocols, hardware]
- [Flag missing data points in the table]

#### Missing Data
- [Methods or papers that should be in this comparison but lack quantitative data in the KB]
- [Suggest which papers to revisit for missing metrics]

## Important
- Convert paper_id underscores to arXiv dots
- Only include numbers that appear explicitly in KB nuggets — do NOT estimate or interpolate
- Use "—" for missing values in the table
- If the same method is evaluated on multiple datasets, include separate rows
- Note when results come from different experimental setups
- End your response with 2-3 suggested follow-up questions under a `### Suggested Follow-ups` heading
