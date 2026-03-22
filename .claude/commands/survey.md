Generate a mini literature review on a topic using the thesis knowledge base.

## Input
Topic to survey: $ARGUMENTS

## Instructions

1. Generate 5-7 diverse sub-queries covering different aspects of the topic (methods, results, limitations, comparisons, background). Run them using the `multi_search` MCP tool with n=50.

2. Also run type-specific searches to ensure coverage using the `semantic_search` MCP tool:
   - `semantic_search(query="$ARGUMENTS", n=20, types=["method"])`
   - `semantic_search(query="$ARGUMENTS", n=20, types=["result", "comparison"])`
   - `semantic_search(query="$ARGUMENTS", n=10, types=["limitation"])`

3. Deduplicate results across all queries (by nugget_id).

4. Cluster the nuggets thematically and organize into a structured review.

## Output Format

### Literature Review: [Topic]

#### Overview
[1-2 paragraph high-level summary of the state of research on this topic]

#### Methods and Approaches
[Organized by approach type. For each approach:]
- **[Approach Name]**: Description with citations (arXiv:XXXX.XXXXX, YEAR). Key characteristics and innovations.

#### Key Results
[Summary of quantitative results, organized by dataset or task:]
- **[Dataset/Task]**: Method X achieved Y% (arXiv:...), while Method Z achieved W% (arXiv:...).

#### Limitations and Open Challenges
[What problems remain unsolved, organized by theme]

#### Trends
[Chronological or thematic trends observed across the surveyed papers]

### Bibliography
1. arXiv:XXXX.XXXXX — "Full Paper Title" (YEAR)
2. ...

### Coverage Gaps
[Topics or aspects that have few or no nuggets in the KB — areas where additional reading may be needed]

## Source Types
Sources are tagged with their nugget type (method, result, comparison, limitation, background, claim).
Use these types to organize your review:
- "method" and "background" sources → Methods and Approaches + Overview sections
- "result" and "comparison" sources → Key Results section
- "limitation" sources → Limitations and Open Challenges section
- "claim" sources → wherever contextually relevant
Ensure each section draws primarily from its corresponding source types.

## Important
- Convert paper_id underscores to arXiv dots
- Cite every factual claim with its source paper
- If multiple papers report similar findings, cite all of them
- Do NOT fabricate connections between papers — only report what the KB evidence shows
- Include the total number of unique papers and nuggets found for this topic
- End your response with 2-3 suggested follow-up questions under a `### Suggested Follow-ups` heading
