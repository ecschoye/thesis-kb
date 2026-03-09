Draft a paragraph for the thesis, grounded in evidence from the knowledge base.

## Input
What to write about: $ARGUMENTS

## Instructions

1. Break the topic into 3-5 sub-queries that cover different aspects. Run them:
   ```
   cd /cluster/work/ecschoye/thesis-kb && source .venv/bin/activate && python -m src.query --queries "sub-query 1" "sub-query 2" "sub-query 3" -n 30 --json
   ```

2. If the topic focuses on specific nugget types, use `--types` to narrow results:
   - Methods/architectures: `--types method`
   - Performance/benchmarks: `--types result,comparison`
   - Open problems: `--types limitation,claim`

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

## Output Format

### Draft Paragraph

[The paragraph text with inline citations]

### Sources
1. arXiv:XXXX.XXXXX — "Full Paper Title" (YEAR)
2. ...

### Notes
- [Any caveats, synthesis notes, or thin-coverage warnings]

## Important
- Convert paper_id format (underscores) to arXiv format (dots): 1510_01972 → arXiv:1510.01972
- Do NOT invent facts — only use evidence from KB results
- If KB coverage is too thin for a full paragraph, say so and output what you can
- Prefer recent papers (2020+) unless older work is foundational