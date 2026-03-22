Fact-check a written paragraph against the thesis knowledge base.

## Input
The text to verify: $ARGUMENTS

## Instructions

1. Parse the input text into individual factual claims. List each claim explicitly.

2. For each claim, search the KB using the `semantic_search` MCP tool with n=10. Run multiple searches in parallel if there are many claims.

3. For each claim, determine its status:
   - **Verified**: A KB nugget directly supports this claim. Include the source.
   - **Partially verified**: Supporting evidence exists but doesn't exactly match (e.g., different numbers, different context).
   - **Unverified**: No relevant evidence found in the KB. This doesn't mean it's wrong — just not in the KB.
   - **Contradicted**: A KB nugget provides conflicting evidence. Include both the claim and the contradicting evidence.

4. For claims with specific numbers (accuracy percentages, latency, FPS, etc.), verify the exact values against KB nuggets.

## Output Format

### Claim-by-Claim Analysis

**Claim 1:** "[extracted claim]"
- Status: **Verified** / **Partially verified** / **Unverified** / **Contradicted**
- Evidence: [nugget text from KB, if found]
- Source: arXiv:XXXX.XXXXX — "Title" (YEAR)
- Notes: [any discrepancies or caveats]

**Claim 2:** ...

### Summary
- X/Y claims verified
- X/Y claims unverified (not in KB)
- X/Y claims contradicted

### Suggested Corrections
[If any claims are contradicted or have wrong numbers, suggest the corrected text with proper citations]

## Important
- Convert paper_id underscores to arXiv dots
- Be precise about numbers — "90% accuracy" vs "89.7% accuracy" matters
- If a claim cites a specific paper, try to find that paper in the KB by searching for its title or key terms
- Do NOT mark something as contradicted unless the KB evidence clearly conflicts — uncertainty is "unverified"
- End your response with 2-3 suggested follow-up questions under a `### Suggested Follow-ups` heading
