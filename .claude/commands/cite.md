Find sources from the thesis knowledge base that support, contradict, or provide context for a claim.

## Input
The user's claim or statement: $ARGUMENTS

## Instructions

1. Run a semantic search against the KB using the claim text:
   ```
   cd /cluster/work/ecschoye/thesis-kb && source .venv/bin/activate && python -m src.query "$ARGUMENTS" -n 20 --json
   ```

2. If the claim has multiple facets, run additional targeted queries using `--queries`:
   ```
   python -m src.query --queries "sub-query 1" "sub-query 2" -n 20 --json
   ```

3. Analyze each returned nugget and classify its relationship to the claim:
   - **Supports**: The nugget directly supports the claim with evidence
   - **Partially supports**: Related evidence that partially backs the claim
   - **Contradicts**: Evidence that contradicts or qualifies the claim
   - **Provides context**: Background information relevant to the claim

4. Group results by paper and format the output as follows:

## Output Format

### Supporting Evidence
For each supporting nugget:
> **[Paper Title]** (arXiv:XXXX.XXXXX, YEAR)
> [The relevant finding from the nugget]
> *Nugget type: [type] | Distance: [dist]*

### Contradicting Evidence
(Same format, if any)

### Contextual Evidence
(Same format, if any)

### Citation Summary
A one-line summary listing all relevant papers for easy copy-paste:
- List each paper as: arXiv:XXXX.XXXXX — "Title" (Year)

## Important
- Always use the full arXiv ID (with dots, e.g., arXiv:1510.01972), converting from the paper_id format (underscores to dots)
- If a paper has no arXiv ID, use its title and year instead
- Rank by relevance (lowest distance = most relevant)
- Skip nuggets with distance > 1.5 as they are likely irrelevant
- Do NOT fabricate citations — only report what the KB returns
- End your response with 2-3 suggested follow-up questions under a `### Suggested Follow-ups` heading