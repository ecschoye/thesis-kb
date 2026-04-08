Describe our implementation of a component, grounded in both codebase knowledge and academic literature.

## Input
Component or module to describe: $ARGUMENTS

## Instructions

1. **Search code nuggets**: Use `code_search` to find implementation details about the component in our SMCM-MCFNet codebase.

2. **Get implementation detail**: Use `implementation_detail` to get a side-by-side view of our code vs. related paper descriptions.

3. **Check cross-references**: Use `find_papers_for_code` to find which papers inspired the implementation.

4. **Get code structure**: Use `code_structure` with a relevant module path to understand the class hierarchy and methods.

5. **Synthesize**: Write a clear, technical description that covers:
   - What the component does and its role in the system
   - Key architectural decisions and why they were made
   - How it relates to methods described in the literature
   - Important parameters and their effects
   - Data flow: inputs, processing, outputs

## Output Format

### Implementation: [Component Name]

**Purpose**: [One-sentence description]

**Architecture**: [Description of classes, inheritance, key methods]

**Design Decisions**:
- [Decision 1]: [Rationale, citing paper if inspired by one]
- [Decision 2]: ...

**Key Parameters**:
| Parameter | Default | Purpose |
|-----------|---------|---------|
| ... | ... | ... |

**Relationship to Literature**:
- Based on: [Paper references]
- Our modifications: [What we changed and why]

### Source Files
- [list of relevant source files with brief descriptions]

## Important
- Reference actual class and method names from the code
- Cite papers using arXiv format (arXiv:XXXX.XXXXX)
- Distinguish between "what the paper proposes" and "what we implemented"
- Note any deviations from the original paper's approach
