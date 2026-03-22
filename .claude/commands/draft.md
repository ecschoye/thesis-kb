Draft a paragraph for the thesis, grounded in evidence from the knowledge base.

## Input
What to write about: $ARGUMENTS

## Mode Detection

Determine which mode to use based on the input:

- **Revise mode**: The input contains an existing draft paragraph (multi-sentence prose, possibly with citations). In this mode, your job is to **improve the existing text**, not rewrite it from scratch.
- **Draft mode**: The input is a topic description, bullet points, or a short prompt without a full draft. In this mode, write a new paragraph.

## Instructions — Revise Mode

1. Extract the key claims and topics from the user's existing draft. Build 2-4 targeted sub-queries and run them using the `multi_search` MCP tool with n=20.

2. Compare the draft against the retrieved evidence. Your role is **editor/advisor**, not ghostwriter. Do NOT output a rewritten paragraph — help the user improve *their* text.

3. For each issue, quote the problematic phrase and suggest a concrete fix inline. Categories:
   - **Factual**: incorrect numbers, unsupported claims — cite KB evidence
   - **Gap**: claims that need a citation, or KB evidence that could strengthen the point
   - **Structure**: redundancy, unclear transitions, logical flow
   - **Style**: awkward phrasing, unnecessary words, clarity

4. Output your response in this format:

### Feedback

For each issue:
- **[Category]** "quoted phrase" → suggested fix or explanation. *(Source if relevant)*

### Sources
1. arXiv:XXXX.XXXXX — "Full Paper Title" (YEAR)
2. ...

### Notes
- [Any caveats, synthesis notes, or thin-coverage warnings]

## Instructions — Draft Mode

1. Break the topic into 3-5 sub-queries that cover different aspects. Run them using the `multi_search` MCP tool with n=30.

2. If the topic focuses on specific nugget types, use the `types` parameter:
   - Methods/architectures: `types=["method"]`
   - Performance/benchmarks: `types=["result", "comparison"]`
   - Open problems: `types=["limitation", "claim"]`

3. Read through all returned nuggets. Identify the key facts, claims, and evidence.

4. Synthesize these into a coherent paragraph with academic tone:
   - Every factual claim must cite its source as (arXiv:XXXX.XXXXX, YEAR)
   - If a paper has no arXiv ID, cite as (Author et al., YEAR) using the paper_authors field
   - Flow logically from general to specific, or chronologically
   - Use hedging language where evidence is from a single source ("X et al. demonstrated that..." rather than "it is known that...")
   - Do not pad with filler — every sentence should carry information

5. After the paragraph, include a **Sources** section listing each cited paper with its full title and arXiv ID.

6. Add a **Notes** section flagging:
   - Claims where you synthesized across multiple papers (mark as "synthesized")
   - Areas where KB coverage is thin and the user should verify independently
   - Specific numbers or metrics cited, so the user can double-check them

### Output Format (Draft Mode)

### Draft Paragraph

[The paragraph text with inline citations]

### Sources
1. arXiv:XXXX.XXXXX — "Full Paper Title" (YEAR)
2. ...

### Notes
- [Any caveats, synthesis notes, or thin-coverage warnings]

## Source Types
Sources are tagged with their nugget type (method, result, comparison, limitation, background, claim).
Use these types to guide your paragraph:
- "method" and "background" sources → technical descriptions and context
- "result" and "comparison" sources → quantitative claims and trade-offs
- "limitation" sources → caveats and hedging language
- "claim" sources → wherever contextually relevant
Ensure claims are grounded in the appropriate source types.

## Important
- Convert paper_id format (underscores) to arXiv format (dots): 1510_01972 → arXiv:1510.01972
- Do NOT invent facts — only use evidence from KB results
- If KB coverage is too thin for a full paragraph, say so and output what you can
- Prefer recent papers (2020+) unless older work is foundational
- In **Revise mode**: preserve the user's writing style and structure. Make surgical edits, not rewrites. If the draft is mostly good, say so and make minimal changes.
- End your response with 2-3 suggested follow-up questions under a `### Suggested Follow-ups` heading
