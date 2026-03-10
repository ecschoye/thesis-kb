Quality review a thesis section using the provided sources as ground truth.

## Instructions

You are given:
1. A `<sources>` block containing KB nuggets retrieved based on claims extracted from the user's text
2. The user's thesis text to review

Using ONLY the provided sources, perform five analyses:

**A. Accuracy** — Verify every factual claim against the sources. Flag wrong numbers, misattributions, or overclaims. Only flag issues where a source directly contradicts the text.

**B. Completeness** — Identify important findings in the provided sources that the text omits. Distinguish between critical omissions (changes the meaning) and nice-to-haves.

**C. Citation quality** — Check that cited papers actually support the claims they're attached to (match citations against source paper names). Flag unsupported or missing citations.

**D. Writing quality** — Assess logical flow, paragraph transitions, redundancy, vague language, and claim strength (overclaiming vs appropriate hedging).

**E. Balance** — Check if the text is biased toward certain methods or papers while ignoring alternatives present in the sources.

## Output Format

### Review: [inferred topic]

#### Accuracy Issues
For each issue:
- **[error/warning/note]** "[quoted text from input]"
  - Problem: [what's wrong]
  - KB evidence: [nugget summary with source paper and arXiv ID]
  - Suggested fix: [corrected text]

If no issues: "No accuracy issues found."

#### Missing Coverage
Findings from the provided sources that should be mentioned:
- **[critical/suggested]** [Paper title] (arXiv:XXXX.XXXXX, YEAR) — [why it matters, with specific detail from the nugget]

If complete: "Coverage is adequate for the scope."

#### Citation Issues
- **[missing/misattributed/unsupported]** "[claim text]" — [explanation]

If no issues: "Citations check out."

#### Writing Feedback
- [Specific, actionable suggestions about structure, flow, hedging, or clarity]

#### Balance Assessment
- [Note any bias toward/against specific approaches, or if coverage is fair]

### Summary
| Category | Score | Issues |
|----------|-------|--------|
| Accuracy | OK / Issues found | X errors, Y warnings |
| Completeness | OK / Gaps found | X critical, Y suggested |
| Citations | OK / Issues found | X issues |
| Writing | OK / Suggestions | X suggestions |
| Balance | OK / Biased | [brief note] |

### Suggested Revision
[If there are accuracy errors or critical omissions, provide a revised version of the problematic paragraphs with corrections inline. Only rewrite paragraphs that need changes. Preserve LaTeX formatting and \cite commands.]

## Source Types
Sources are tagged with their nugget type (method, result, comparison, limitation, background, claim).
Use type metadata to assess whether the text adequately covers each dimension of the topic.

## Important
- Be specific — quote the exact text that has issues and provide exact replacements
- Distinguish between errors (factually wrong backed by source evidence) and suggestions (could be improved)
- Do NOT invent issues — only flag problems where a provided source contradicts or supplements the text
- Preserve the author's voice, style, and LaTeX formatting in suggested revisions
- When a source nugget provides a useful quantitative detail (e.g. "6.9 mAP drop"), recommend incorporating it with a specific suggestion
- End your response with 2-3 suggested follow-up questions under a `### Suggested Follow-ups` heading
