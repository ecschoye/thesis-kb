Generate Architecture chapter prose for a module, combining literature context with code implementation details.

## Input
Module or subsection to describe: $ARGUMENTS

## Instructions

1. **Retrieve code knowledge**: Use `code_search` and `code_structure` to understand our implementation.

2. **Retrieve paper knowledge**: Use `multi_search` with queries about the methods our implementation is based on. Use types=["method", "background"] for architectural context.

3. **Check cross-references**: Use `find_papers_for_code` to identify which papers our code implements or extends.

4. **Get training configs**: If relevant, use `training_config` to pull hyperparameter details.

5. **Write thesis-quality prose**:
   - Academic tone, third person ("The system employs..." not "We use...")
   - Cite sources for any method adopted from literature: \citep{Author_YEAR:slug}
   - Clearly distinguish novel contributions from adopted methods
   - Include mathematical notation where appropriate (LaTeX math mode)
   - Reference figures/tables if they exist in the thesis
   - Follow the supervisor's rules: no first person, no vague referents, no unhedged claims

## Output Format

### [Section Title]

[2-4 paragraphs of thesis-quality prose]

Each paragraph should:
- Open with the component's purpose in the overall pipeline
- Describe the architecture with reference to class names and data flow
- Cite adopted methods from literature
- Highlight novel modifications or contributions
- Include quantitative details (dimensions, parameter counts) where available

### LaTeX-Ready Citations
```
\citep{Author_YEAR:slug} — Paper Title (YEAR)
```

### Notes
- [Caveats, areas needing verification, thin coverage]

## Important
- This is for the Architecture chapter — focus on WHAT and HOW, not results
- Do NOT include experimental results or comparisons (those go in Experiments chapter)
- Follow chapter boundaries: Architecture = our system design only
- Every adopted method needs a citation
- Novel contributions should be clearly marked as such
