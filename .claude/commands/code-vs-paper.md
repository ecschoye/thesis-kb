Compare our implementation against a paper's described method.

## Input
Paper identifier or method name: $ARGUMENTS

## Instructions

1. **Identify the paper**: Use `find_papers` or `get_paper_info` to locate the paper in the KB.

2. **Get paper's method description**: Use `get_paper_nuggets` with types=["method"] to retrieve how the paper describes the approach.

3. **Find our implementation**: Use `find_implementations` with the paper_id to discover which code files implement or extend this paper's methods.

4. **Get code details**: Use `code_search` and `implementation_detail` to retrieve our implementation knowledge.

5. **Compare systematically**:

## Output Format

### Comparison: [Paper Title] vs. Our Implementation

**Paper**: [Full title, authors, year, arXiv ID]
**Code**: [Source file paths]

| Aspect | Paper's Description | Our Implementation | Difference |
|--------|--------------------|--------------------|------------|
| Architecture | ... | ... | ... |
| Loss function | ... | ... | ... |
| Training strategy | ... | ... | ... |
| Key parameters | ... | ... | ... |
| Data representation | ... | ... | ... |

### Adopted As-Is
- [Components we took directly from the paper]

### Modified
- [Components we changed, with rationale]

### Novel Additions
- [Components not in the original paper]

### Implications for Thesis
- [How these differences should be discussed in the Architecture/Experiments chapters]
- [Claims that need careful wording due to modifications]

## Important
- Be precise about what's the same vs. different
- When our implementation diverges, note whether it's intentional design or practical constraint
- Use arXiv citation format for the paper
- Reference exact class/method names from our code
