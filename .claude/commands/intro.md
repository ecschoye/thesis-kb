Generate a chapter introduction paragraph for the thesis.

## Input
Chapter to introduce: $ARGUMENTS

## Instructions

1. **Resolve the chapter file.** The argument is either:
   - A chapter number (e.g., `4`) — map to `~/TDT4900-master-thesis/chapters/{N}-*.tex` by globbing
   - A filename (e.g., `4-background.tex`) — use directly under `~/TDT4900-master-thesis/chapters/`
   - A full path — use as-is

2. **Read the target chapter file** from `~/TDT4900-master-thesis/chapters/`. Extract:
   - The `\chapter{}` title
   - All `\section{}` and `\subsection{}` headings (with their `\label{}`s)
   - Any existing introduction text (content before the first `\section{}`)
   - The general scope of each section (skim the first few lines or paragraph of each)

3. **Read the introduction chapter** (`~/TDT4900-master-thesis/chapters/3-introduction.tex`) to understand:
   - The thesis goal and problem statement
   - The research questions (RQs)
   - The overall narrative arc and chapter roadmap (if present)

4. **Read the adjacent chapters** (the chapter before and after the target) to understand:
   - What context the reader arrives with
   - What the current chapter needs to set up for the next one

5. **Draft the introduction paragraph.** It should:
   - Open by motivating why this chapter's topic matters in the context of the thesis
   - Connect to what the previous chapter established (use `\Cref{}` for back-references)
   - Preview the sections in order, briefly stating what each covers and how they build on each other
   - Close by noting what understanding the reader will have by the end, and how it feeds into the next chapter
   - Use academic tone — precise, concise, no filler phrases like "it is worth noting" or "in recent years"
   - Be 4-8 sentences long (one substantial paragraph)
   - Use `\Cref{}` for all cross-references (chapters, sections, figures)

6. **Do NOT query the knowledge base.** Chapter introductions are structural and narrative — they describe what is in the chapter and why, not evidence from papers.

## Output Format

### Chapter Introduction: [chapter title]

```latex
[The LaTeX paragraph, ready to paste after \chapter{} and before the first \section{}]
```

### Structural Notes
- **Incoming context:** [what the previous chapter covered]
- **Outgoing setup:** [what the next chapter expects from this one]
- **Section flow:** [1-2 sentences on how the sections connect]

### Suggestions
- [Any structural observations, e.g., missing transitions, sections that could be reordered, or content that might belong in a different chapter]
