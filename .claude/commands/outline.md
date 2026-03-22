Generate a thesis section outline based on available evidence in the knowledge base.

## Input
Section to outline: $ARGUMENTS

## Instructions

1. Generate 5-8 sub-queries spanning the likely scope of this section. Run them using the `multi_search` MCP tool with n=50.

2. Run type-stratified queries using the `semantic_search` MCP tool:
   - `semantic_search(query="[main topic]", n=15, types=["background"])`
   - `semantic_search(query="[main topic]", n=20, types=["method"])`
   - `semantic_search(query="[main topic]", n=15, types=["result", "comparison"])`
   - `semantic_search(query="[main topic]", n=10, types=["limitation"])`

3. Cluster the nuggets thematically. Identify natural groupings.

4. Propose a section structure that follows the evidence.

## Output Format

### Outline: [Section Title]

#### Proposed Structure

**[Section Number] [Section Title]**
- Key points to cover:
  - [Point 1] — supported by N nuggets from [Paper A (YEAR), Paper B (YEAR)]
  - [Point 2] — supported by N nuggets from [...]
- Evidence density: **Strong** / **Moderate** / **Thin**

**[Section Number] [Subsection Title]**
- Key points to cover:
  - ...
- Evidence density: ...

[Repeat for each proposed subsection]

#### Suggested Flow
[Brief description of the recommended ordering and why — chronological, by complexity, by approach type, etc.]

#### Strong Coverage Areas
[Topics where the KB has 10+ relevant nuggets across multiple papers]

#### Thin Coverage Areas
[Topics where the KB has <3 nuggets — may need additional reading or different framing]

#### Key Papers for This Section
1. arXiv:XXXX.XXXXX — "Title" (YEAR) — contributes to [which subsections]
2. ...

## Source Types
Sources are tagged with their nugget type (method, result, comparison, limitation, background, claim).
Use these types to assess section viability:
- "method" and "background" sources → indicate strong coverage for technical sections
- "result" and "comparison" sources → indicate data for evaluation sections
- "limitation" sources → indicate material for discussion/future work sections
- "claim" sources → wherever contextually relevant
Evidence density ratings should reflect source type distribution, not just raw counts.

## Important
- Convert paper_id underscores to arXiv dots
- Evidence density ratings should reflect actual nugget counts, not guesses
- Suggest only structures that the KB evidence can support
- If a section the user expects to write has thin KB coverage, say so explicitly
- End your response with 2-3 suggested follow-up questions under a `### Suggested Follow-ups` heading
