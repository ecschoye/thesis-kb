End-of-session update. Do all of the following:

1. **Devlog**: Prepend a new entry to ~/vault/hippocampus/devlog/writing.md (newest first). Template:

   ---
   ## YYYY-MM-DD
   **What changed:**
   - (2-4 bullets: sections written, citations added, figures updated — be specific)
   **Decisions:**
   - (structural/content decisions and WHY. Link to decision file if created. Skip section if none.)
   **Tried but didn't work:**
   - (failed approaches and why. Skip section if none.)
   **Next up:**
   - (1-3 concrete actionable steps)
   ---

   Max 8 lines of content per entry.

2. **Status**: Overwrite ~/vault/prefrontal/status/writing-status.md with current state:

   # Writing Status
   **Last updated:** YYYY-MM-DD
   ## Current focus
   (which chapter/section)
   ## Chapters status
   - Introduction: (draft/in progress/not started)
   - Background Theory: (draft/in progress/not started)
   - State of the Art: (draft/in progress/not started)
   - Method: (draft/in progress/not started)
   - Experiments: (draft/in progress/not started)
   - Results: (draft/in progress/not started)
   - Discussion: (draft/in progress/not started)
   - Conclusion: (draft/in progress/not started)
   ## Recent milestones
   (last 3-5 completed items)
   ## Blockers
   (anything stuck)

3. **Decisions**: If any significant decisions were made, create individual files in ~/vault/cortex/decisions/ named YYYY-MM-DD-short-description.md. Template:

   # Decision Title
   **Date:** YYYY-MM-DD
   **Scope:** impl | writing | both
   **Status:** active
   ## Context
   (what prompted this)
   ## Decision
   (what was decided)
   ## Rationale
   (why)
   ## Implications
   - Implementation: (what changes)
   - Thesis: (what this means for writing)
   ## Related
   - [[concept-links]]

4. **Concepts**: If you learned or clarified anything about a concept, update the relevant file in ~/vault/cortex/concepts/. Add to Notes section. Add new [[links]] to other concepts.

5. **Cross-project**: If anything affects the implementation, also prepend an entry to ~/vault/hippocampus/devlog/shared.md.

Show me a summary of all changes before saving. Do not commit to git.
