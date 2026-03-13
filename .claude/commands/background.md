Draft or revise a background theory paragraph — neutral, textbook-style writing grounded in evidence from the knowledge base.

## Input
What to write about: $ARGUMENTS

## Pre-fetched Cited Papers

If the input contains `\cite{...}` keys, a hook has already looked them up in the KB. The results appear in your context as "Pre-fetched Cited Papers (from hook)". **Use those results directly** — do NOT re-query these papers with `--find-author` or semantic search. Only mark a paper as "Missing from KB" if the hook result says "Not found in KB" for that key.

---

## HARD RULES — Read Before Anything Else

These three rules are **non-negotiable**. Violating any of them is a critical failure.

1. **Citation keys**: NEVER use short-form keys like `\cite{hamann2024}`, `\cite{zhu2018}`, or `\cite{smith2023}`. Every `\cite{}` must be either (a) the user's exact key from their input, or (b) a new key in `\cite{AuthorLast_YEAR:descriptive-slug}` format marked `[new key — verify]`. Example: `\cite{Hamann_2024:event-tracking-any-point}` — NOT `\cite{hamann2024}`.

2. **No LaTeX in metadata**: The ONLY place LaTeX (`$...$`, `\rightarrow`, `\textbf{}`) may appear is inside the ```latex paragraph block. Sources, Notes, Metric Verification, Covered so far, Outline, and Next sections use **plain text and markdown only**. Write `C` not `$C$`, write `(x, y, p, t)` not `$(x, y, p, t)$`, write `|delta log I| >= C` not `$|\Delta \log I| \geq C$`.

3. **No comparisons in paragraph text**: NEVER use "unlike", "in contrast to", "compared to", "whereas traditional", or mention any technology not being defined in the current paragraph. If writing about event cameras, do not mention frame cameras. Delete comparison clauses entirely — do not soften them.

4. **No raw output**: Your response must contain ONLY the structured sections (Outline, Paragraph, Sources, Metric Verification, Notes, Next/All Sources/Section Summary, Suggested Follow-ups). NEVER include query strings, distance scores, paper titles with `dist=`, JSON, or any retrieval metadata. If any such content appears in your response, delete it before responding.

5. **Missing from KB**: NEVER report a paper as "Missing from KB" if the hook's Status Summary says **FOUND** for that key. The hook has already looked it up — the data is in your context. Only report papers as missing if the Status Summary explicitly says **NOT FOUND**.

## Principles

This mode produces objective, definitional writing. The goal is to *define and explain*, not to argue, compare, or motivate.

- **Neutral tone**: Describe properties as facts, not as strengths or weaknesses. Write "CMOS sensors provide a dynamic range of approximately 60 dB" — not "sensors suffer from a limited dynamic range."
- **No comparisons**: Do not introduce other technologies as contrast, even implicitly. If the topic is frame-based cameras, do not mention event cameras. Each concept stands on its own terms.
- **No advocacy**: Avoid loaded language: "suffer from", "limited", "only", "merely", "pales in comparison", "significant drawback", "unfortunately". State trade-offs neutrally: "reducing exposure time decreases motion blur but lowers signal-to-noise ratio."
- **KB bias awareness**: The knowledge base is dominated by papers that argue for event cameras / SNNs over conventional approaches. Many retrieved nuggets will frame conventional methods negatively. Actively strip this framing — extract only the factual content from such nuggets.
- **Stay on topic**: Discard retrieved nuggets about adjacent topics. If the paragraph is about X, do not introduce Y.
- **Textbook voice**: Write as if explaining to a knowledgeable reader who has no agenda. Define terms precisely, explain mechanisms clearly, cite measurable properties.

## Scope Detection

Before doing anything else, classify the input scope:

1. **Single paragraph** — a topic, bullet points, or short prompt about one concept → draft/revise one paragraph
2. **Multi-paragraph** — input contains multiple `\paragraph{}` headings, or bullet groups separated by clear topic breaks → process as a **paragraph sequence**
3. **Subsection/Section** — input contains `\subsection{}` or `\section{}` with multiple paragraphs → process as a **structured section**

### Behavior by scope:

- **Single paragraph**: Draft or revise exactly one paragraph (original behavior).
- **Multi-paragraph / Subsection / Section**:
  1. List all paragraphs detected in the input with their headings/topics
  2. Draft the **first paragraph only**, with full sources and notes
  3. End with a `### Next` prompt telling the user which paragraph comes next, and ask them to say "continue" (or provide feedback on the current paragraph first)
  4. When the user says "continue", draft the next paragraph. Repeat until all paragraphs are done.
  5. After the final paragraph, output a consolidated `### All Sources` list and a `### Section Summary` with word count and coverage notes.

This iterative approach ensures each paragraph gets proper KB retrieval and quality, while allowing the user to steer, reorder, or skip paragraphs.

## Mode Detection

- If the input contains an existing draft paragraph (multi-sentence prose): **revise** it — preserve structure and voice, make surgical corrections only.
- If the input is a topic, bullet points, or short prompt: **write** a new paragraph from scratch.
- If the input is a mix (some paragraphs drafted, some as bullet points): apply the appropriate mode per paragraph in the iterative flow.

## Process

### Step 1: Use Pre-fetched Cited Papers

A hook has already searched the KB for every `\cite{}` key in the input. The results are injected into your context as **"Pre-fetched Cited Papers (from hook)"** with a **Status Summary** at the top showing FOUND/NOT FOUND for each key.

**CRITICAL**: Read the Status Summary. For every key marked **FOUND**, the hook has provided the paper_id, title, authors, and up to 30 nuggets. Use this data directly — it IS the KB lookup result.

Rules:
- A paper is "Missing from KB" ONLY if the Status Summary says **NOT FOUND** for that key.
- If Status Summary says **FOUND**, the paper IS in the KB. Do NOT mark it as missing. Do NOT re-query it.
- Do NOT run `--find-author` or semantic search for any pre-fetched paper.

### Step 2: Semantic Search (additional supporting literature)

Detect environment: if `/cluster/work/ecschoye/thesis-kb` exists use `cd /cluster/work/ecschoye/thesis-kb && source .venv/bin/activate`; otherwise `cd ~/thesis-kb && source venv/bin/activate`.

**Only run this AFTER reviewing Step 1 results.** This finds papers the user did NOT cite:
```
python -m src.query --queries "topic query 1" "topic query 2" -n 15 --json
```

### Step 3: Build Citation Key Registry

After Steps 1 and 2, build an explicit mapping table from user-supplied cite keys to KB results. **Start by copying every FOUND entry from the hook's Status Summary** — these are already resolved:

```
Citation Key Registry:
  \cite{Gallego_2022:event-based-vision}  → paper_id: 1904_08405 ✓ hook-found
  \cite{Cordone_2022:SNN-obj-det}         → paper_id: 2206_06506 ✓ hook-found
  \cite{Unknown_2025:something}           → ✗ hook: NOT FOUND
```

**CRITICAL — Citation key rules:**
- For papers the user cited: use the user's EXACT `\cite{...}` key. NEVER rename, shorten, or reinvent keys. `\cite{Gallego_2022:event-based-vision}` stays exactly that — not `\cite{gallego2022}`, not `\cite{Gallego2022}`.
- For papers found via general retrieval that the user did NOT cite: create a key in `\cite{AuthorLast_YEAR:short-slug}` format and mark it `[new key — verify]` in the Sources section so the user can confirm or replace it.
- NEVER invent short-form keys like `\cite{hamann2024}` or `\cite{chakravarthi2025}`.

Carry this registry through all subsequent steps. Every `\cite{}` in the output must trace back to either the registry or a `[new key — verify]` entry.

### Step 4: Extract Metric Claims from Bullets

Scan the input bullet points for specific numbers, measurements, and technical claims. These are **claims to verify**, not just topic hints.

Examples of extractable claims:
- "120–140 dB" → query: `"event camera dynamic range 120 dB 140 dB"`
- "pixel sizes: 18.5 um (DAVIS346), 9 um (DVXplorer)" → query: `"DAVIS346 pixel size" "DVXplorer pixel pitch"`
- "2 MHz to 1200 MHz readout" → query: `"AER readout rate MHz event camera"`

Run metric verification queries:
```
python -m src.query --queries "event camera dynamic range dB" "DAVIS346 pixel size micron" -n 10 --json --types background,method,result
```

For each metric found in the input, track whether KB evidence **confirms**, **contradicts**, or **has no data** for that value. Report this in the Notes section.

### Step 5: General Topic Retrieval

Build 2-4 sub-queries focused on the *mechanics and properties* of the topic (not comparisons, not limitations-of-X-vs-Y):
```
python -m src.query --queries "sub-query 1" "sub-query 2" -n 20 --json --types background,method
```

### Step 6: Filter and Merge

Merge results from Steps 1-5, deduplicate by paper_id, then filter ruthlessly:
- Keep only nuggets that describe *how the topic works* or *what its properties are*
- Discard nuggets that exist primarily to compare, criticize, or motivate an alternative technology
- Discard nuggets with evaluative framing ("X outperforms", "unlike traditional", "addresses the limitations of")
- From comparative nuggets, extract only the factual claim about the topic at hand if one exists
- **Prioritize**: anchor papers (Step 1) > metric-confirming nuggets (Step 4) > general topic nuggets (Step 5)

### Step 7: Assign Confidence Levels

For each factual claim that will appear in the output, assess evidence strength:

- **Strong** (3+ independent sources, or from a survey/review paper): state as fact
- **Moderate** (2 sources, or 1 well-cited recent paper): state with light attribution — "X et al. report that..."
- **Weak** (single non-survey source, old paper, or nugget type is "claim" rather than "result"/"method"): use hedged language — "according to X et al.," or "X et al. observed that..." — and flag in Notes as `⚠ single-source`

Mark confidence inline in the Sources section: `[strong]`, `[moderate]`, or `[weak]`.

### Step 8: Draft or Revise

**If revising** an existing draft:
- Your role is **editor/advisor**, not ghostwriter. Help the user improve *their* paragraph.
- Give specific, actionable feedback that the user can apply themselves
- For each issue, quote the problematic phrase and suggest a concrete fix inline
- Do NOT output a rewritten paragraph. The user writes; you advise.
- Categories of feedback:
  - **Bias/tone**: loaded language, implicit comparisons, advocacy framing — quote the phrase, explain why it's biased, suggest a neutral alternative
  - **Factual**: incorrect numbers, unsupported claims — cite the KB evidence that contradicts or supports
  - **Gaps**: claims that need a citation, or where KB evidence could strengthen the point
  - **Structure**: redundancy, unclear transitions, logical flow issues
  - **Style**: awkward phrasing, unnecessary words, clarity improvements

**If writing** a new paragraph (no existing draft provided):
- Define the concept, explain the mechanism, state measurable properties with citations
- **Citation keys**: Look up every citation in the Citation Key Registry (Step 3). Use the user's exact `\cite{}` key for registered papers. For unregistered papers, use `\cite{AuthorLast_YEAR:short-slug}` and mark `[new key — verify]` in Sources. NEVER invent short-form keys.
- After composing each sentence, verify it describes ONLY the topic being defined. If it references another technology (frame cameras, traditional sensors, etc.), delete the comparison clause and rewrite the sentence to stand alone. Do not soften comparisons — remove them entirely.
- Flow logically: definition → mechanism → properties → trade-offs
- Do not pad with filler — every sentence should carry information
- Apply confidence-appropriate language (Step 7)

### Step 9: Final Validation

Before outputting your response, perform these four checks. If any fails, fix the violation before responding.

1. **Citation key audit**: Scan every `\cite{...}` in the paragraph text. For each one:
   - Is it an exact match to a user-supplied key? → OK
   - Is it in `AuthorLast_YEAR:descriptive-slug` format AND marked `[new key — verify]` in Sources? → OK
   - Is it a short-form key like `\cite{smith2023}` or `\cite{hamann2024}`? → **VIOLATION**. Rewrite the key to `\cite{AuthorLast_YEAR:descriptive-slug}` format immediately.

2. **LaTeX leak audit**: Scan every section OUTSIDE the ```latex block. If you find `$`, `\rightarrow`, `\textbf`, `\geq`, `\Delta`, `\in`, `\{`, or any LaTeX command → **VIOLATION**. Replace with plain text equivalent.

3. **Banned phrase audit**: Scan every sentence in the ```latex paragraph block for these patterns:
   - "unlike", "in contrast to", "compared to", "whereas", "conventional", "traditional" → **VIOLATION**. Delete the comparison clause entirely and rewrite the sentence to describe the topic on its own terms. Example: "Unlike frame-based sensors, these devices provide asynchronous output" → "The sensor provides asynchronous output without a global clock or fixed frame rate."
   - "limited", "poor", "suffers from", "fails to", "only", "merely" → **VIOLATION**. State the property neutrally.
   - Any mention of a technology NOT being defined in this paragraph → **VIOLATION**. Remove entirely.

4. **Output cleanliness audit**: Scan for any text after `### Suggested Follow-ups` content, or any lines containing `dist=`, `0.0`, paper titles with floating-point scores, raw query strings, or JSON. → **VIOLATION**. Delete the offending content.

5. **Missing from KB audit**: Scan your Notes section for any "Missing from KB" entries. For each one, check the hook's Status Summary in your context. If the Status Summary says **FOUND** for that key → **VIOLATION**. Remove it from the "Missing" list — the paper IS in the KB and you have its data.

## Cross-Paragraph Coherence (Multi-Paragraph Mode)

When processing multiple paragraphs iteratively, maintain a running **Covered Concepts** list with this exact structure:

```
**Covered so far**:
- **Terms defined**: event tuple (x, y, t, p), contrast threshold C, polarity p in {-1, +1}
- **Equations**: |delta log I| >= C
- **Metrics**: dynamic range 120–140 dB, sub-ms latency, microsecond timestamps
- **Papers cited**: \cite{Gallego_2022:event-based-vision}, \cite{Cordone_2022:SNN-obj-det}
```

**REMINDER**: This section is plain markdown. Do NOT use `$...$` or any LaTeX syntax here. Write `C` not `$C$`, write `(x, y, t, p)` not `$(x,y,t,p)$`.

List **individual** terms, equations, and metric values — not topic-level summaries. This must be granular enough that the next paragraph can check whether a specific symbol, number, or definition has already been introduced.

Before drafting each subsequent paragraph, check the list to:
- Avoid re-defining a term already introduced (reference it instead)
- Avoid repeating the same metric or fact
- Avoid contradicting a previous paragraph's framing
- Use brief back-references when needed: "The contrast threshold C introduced above..."

Display the running list in each iteration's Notes section.

## Bias Checklist — BANNED PHRASES

Before finalizing, scan every sentence for these **banned patterns**. If found, do NOT soften them — **delete the clause entirely** and rewrite the sentence to describe the topic on its own terms.

**Banned — delete on sight:**
- Comparative phrases: "unlike", "in contrast to", "compared to", "whereas traditional", "conventional X fails to"
- Loaded adjectives: "limited", "poor", "insufficient", "mere", "only", "just"
- Advocacy framing: "suffers from", "is constrained by", "fails to", "struggles with", "is plagued by"
- Out-of-scope technology: any mention of a technology not being defined in THIS paragraph (e.g., if defining event cameras, do not mention frame cameras)
- Unsupported superlatives: "major challenge", "significant bottleneck", "critical limitation"

**How to fix (delete, don't soften):**
- ~~"Unlike frame-based systems, event cameras provide asynchronous output"~~ → "Event cameras provide asynchronous output without a global clock or fixed frame rate"
- ~~"limited dynamic range of 60 dB"~~ → "a dynamic range of approximately 60 dB"
- ~~"suffers from motion blur"~~ → "exhibits motion blur when objects move during the exposure interval"
- ~~"traditional sensors fail in HDR scenes"~~ → "sensor response saturates in regions where scene irradiance exceeds the sensor's dynamic range"

## Output Hygiene

Your response MUST contain ONLY the structured sections defined in Output Format below. **NEVER include** any of the following in your response:
- Raw query output, distance scores, `dist=` lines, or JSON dumps
- Query strings or `python -m src.query` commands
- Intermediate working notes, reasoning traces, or nugget text
- Paper titles with scores or retrieval metadata
- Concatenated query text without spaces or formatting

Process all query results **silently** — use them to compose the output sections, then discard. Always use the `--json` flag on all `python -m src.query` commands.

**Your response ENDS after `### Suggested Follow-ups`.** If any text appears after that heading's content, delete it. No trailing output of any kind.

## Format Consistency Rule

The paragraph body uses LaTeX when input is LaTeX-structured. All other sections (Sources, Metric Verification, Notes, Next, Outline, Covered so far) use **plain markdown only**. Do NOT embed LaTeX commands (`$...$`, `\rightarrow`, `\textbf{}`, etc.) in metadata sections. Use plain arrows (`→`), plain text, and markdown formatting.

**Paper titles may contain LaTeX** (e.g., `sub-$\mu$W/Channel`). When writing paper titles in Sources or Notes, strip all LaTeX: replace `$\mu$` with `μ`, `$\rightarrow$` with `→`, remove `$...$` delimiters, etc. Output the plain-text version of the title.

## Output Format — Revising

When the user provides an existing draft, do NOT rewrite it. Output feedback only:

### Feedback

For each issue, use this format:
- **[Category]** "quoted phrase from draft" → suggested fix or explanation. *(Source if relevant)*

Example:
- **Bias** "suffer from a limited dynamic range" → "exhibit a dynamic range of approximately 60 dB" — the original implies a deficiency; state the property neutrally.
- **Gap** "22 to 33 ms where no motion is captured" → this claim needs a citation. Supported by (Gehrig and Scaramuzza, 2024).
- **Redundancy** The frame-rate/throughput trade-off is stated in sentences 5 and 10 — consolidate into one.
- **Factual** "typically around 60 dB" → KB confirms 60 dB for standard CMOS (Lakshmi et al., 2019), but high-end sensors reach up to 95 dB — consider adding this range.

### Sources
1. arXiv:XXXX.XXXXX — "Full Paper Title" (YEAR) `[confidence]`
2. ...

### Notes
- [Caveats, thin coverage, metrics to verify]

## Output Format — Drafting (Single Paragraph)

When no existing draft is provided and scope is a single paragraph:

### Paragraph

If the input contained LaTeX markup (`\paragraph{}`, `\cite{}`, `\subsection{}`), output in LaTeX.

**Line wrapping**: Break lines at ~80 characters. Insert a newline after natural phrase boundaries (after a comma, period, or before a new clause). NEVER output the entire paragraph as a single long line. Each sentence should span 1-3 lines. Example:

```latex
Each pixel independently monitors the logarithmic intensity
of incident light and triggers an event when the change exceeds
a contrast threshold~$C$, typically
10--15\%~\cite{Gallego_2022:event-based-vision,
  Cordone_2022:SNN-obj-det}.
The sensor operates asynchronously without a global clock,
enabling sub-millisecond transmission
latency~\cite{Gallego_2022:event-based-vision}.
```

Otherwise output in markdown with inline citations as `(arXiv:XXXX.XXXXX, YEAR)`.

### Sources
1. arXiv:XXXX.XXXXX — "Full Paper Title" (YEAR) `[confidence]` → `\cite{Gallego_2022:event-based-vision}` (user key)
2. arXiv:XXXX.XXXXX — "Full Paper Title" (YEAR) `[confidence]` → `\cite{Hamann_2024:event-tracking-any-point}` [new key — verify]
NEVER: `\cite{hamann2024}`, `\cite{gallego2022}` — these short-form keys are forbidden.

### Metric Verification
| Claim | Input Value | KB Value | Status | Source |
|-------|------------|----------|--------|--------|
| Dynamic range | 120–140 dB | 120 dB (base 86 dB) | ⚠ nuanced | Chung et al. 2024 |
| Pixel size DAVIS346 | 18.5 µm | 18.5 µm | ✓ confirmed | Shariff et al. 2024 |

*(Only shown when input contained specific metrics to verify. Omit if no metrics in input.)*

### Notes
- [Caveats, thin coverage, metrics to verify]
- ⚠ single-source claims: [list any]
- 🔍 Missing from KB: [any \cite{} keys not found]

## Output Format — Drafting (Multi-Paragraph / Section)

When scope is multi-paragraph or section, output for each iteration:

### Outline
*(First iteration only)* List all detected paragraphs with their mode:
1. `\paragraph{Name}` — brief topic summary *(N citations)* **[draft]**
2. `\paragraph{Name}` — brief topic summary *(N citations)* **[revise]**
3. ...

Use **[draft]** if the user provided bullets/topic only. Use **[revise]** if the user provided existing multi-sentence prose.

### Paragraph N: [Name]

```latex
[The paragraph text in LaTeX with \cite{} keys.
 Break lines at ~80 characters — see line wrapping
 rule in Single Paragraph format above.]
```

### Sources
1. arXiv:XXXX.XXXXX — "Full Paper Title" (YEAR) `[confidence]` → `\cite{Gallego_2022:event-based-vision}` (user key)
2. arXiv:XXXX.XXXXX — "Full Paper Title" (YEAR) `[confidence]` → `\cite{Hamann_2024:event-tracking-any-point}` [new key — verify]
NEVER: `\cite{hamann2024}` — short-form keys are forbidden.

### Metric Verification
*(If applicable — see single-paragraph format above)*

### Notes
- [Caveats, thin coverage, metrics to verify]
- ⚠ single-source claims: [list any]
- 🔍 Missing from KB: [any \cite{} keys not found]
- **Covered so far**: (use structured format from Cross-Paragraph Coherence section — plain text only, NO LaTeX)

### Next
Up next: **\paragraph{Name}** — [brief description]. Say "continue" to proceed, or provide feedback on the current paragraph first.

*(After the final paragraph, replace the Next section with:)*

### All Sources
[Consolidated, deduplicated source list across all paragraphs with bib keys]

### Section Summary
- Total paragraphs: N
- Estimated word count: ~X
- Coverage notes: [any gaps, thin areas, or paragraphs that need more KB support]
- Cross-paragraph issues: [any redundancy or consistency concerns detected]

## Important
- Convert paper_id format (underscores) to arXiv format (dots): 1510_01972 → arXiv:1510.01972
- Do NOT invent facts — only use evidence from KB results
- If KB coverage is too thin for a full paragraph, say so and output what you can
- Prefer recent papers (2020+) unless older work is foundational
- When user supplies `\cite{}` keys, use those exact keys in output — do not rename them
- End your response with 2-3 suggested follow-up questions under a `### Suggested Follow-ups` heading

## STOP

Your response ENDS after `### Suggested Follow-ups`. Do NOT output anything after that section — no query results, no paper lists, no retrieval metadata, no debug output. If you see tool call results in your context, do NOT echo or reproduce them in your response. STOP after Suggested Follow-ups.
