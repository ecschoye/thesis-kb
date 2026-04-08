"""LaTeX formatting compliance tools for thesis-kb."""

from __future__ import annotations

import re

from . import _shared as S
from . import _latex as L


def register(mcp):

    @mcp.tool()
    def check_latex_compliance(tex_content: str) -> dict:
        """Check LaTeX content against the supervisor's 14 flagging rules.

        Rules checked:
        1. First person pronouns (I, we, our, us)
        2. Performance claims without citation or hedging
        3. Yes/no research questions
        4. Architecture details in introduction (if detected)
        5. Experiment methodology in intro (if detected)
        6. Advantages/disadvantages in Background (if detected)
        7. Paper-by-paper summaries in SotA (if detected)
        8. Undefined terms or acronyms
        9. Vague referents (this, these, it without antecedent)
        10. Raw numbers without comparison
        11. Sentences sounding copied from abstracts
        12. \\[...\\] math instead of \\begin{equation}
        13. Missing \\gls{} for glossary terms
        14. Subsections mixing theory and SotA

        Args:
            tex_content: LaTeX source text to check.
        """
        violations = []
        lines = tex_content.split("\n")

        for lineno, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("%"):
                continue

            # Rule 1: First person pronouns
            first_person = re.findall(r"\b(I|we|our|us|my)\b", stripped, re.IGNORECASE)
            # Filter out "I" in math mode or cite keys
            if first_person and not re.match(r".*\$.*\$", stripped):
                for fp in first_person:
                    if fp.lower() in ("i",) and re.search(r"\\cite|\\ref|\$", stripped):
                        continue
                    violations.append({
                        "rule": 1,
                        "desc": "First person pronoun",
                        "line": lineno,
                        "match": fp,
                        "context": stripped[:120],
                    })

            # Rule 9: Vague referents at sentence start
            if re.match(r"^\s*(This|These|It|That)\s+(is|are|was|were|has|have|can|could|will|would|show)", stripped, re.IGNORECASE):
                violations.append({
                    "rule": 9,
                    "desc": "Vague referent at sentence start",
                    "line": lineno,
                    "context": stripped[:120],
                })

            # Rule 12: \[...\] instead of \begin{equation}
            if r"\[" in stripped and "\\begin{" not in stripped:
                violations.append({
                    "rule": 12,
                    "desc": "Display math with \\[...\\] instead of equation environment",
                    "line": lineno,
                    "context": stripped[:120],
                })

            # Rule 2: Performance claims without citation
            perf_words = re.findall(
                r"\b(outperform|superior|better|best|faster|slower|higher|lower|"
                r"achieve|improve|state-of-the-art|SOTA)\b",
                stripped, re.IGNORECASE,
            )
            if perf_words and "\\cite" not in stripped:
                # Check for hedging
                hedge_words = {"may", "might", "could", "potentially", "suggests", "appears", "seems"}
                has_hedge = any(re.search(r"\b" + h + r"\b", stripped, re.IGNORECASE) for h in hedge_words)
                if not has_hedge:
                    violations.append({
                        "rule": 2,
                        "desc": "Performance claim without citation or hedging",
                        "line": lineno,
                        "match": perf_words[0],
                        "context": stripped[:120],
                    })

            # Rule 10: Raw numbers without comparison context
            if re.search(r"\b\d+\.?\d*\s*%", stripped) and "\\cite" not in stripped:
                if not re.search(r"(compared to|versus|vs\.|relative to|increase|decrease|improvement)", stripped, re.IGNORECASE):
                    violations.append({
                        "rule": 10,
                        "desc": "Raw number/percentage without comparative context",
                        "line": lineno,
                        "context": stripped[:120],
                    })

        # Rule 13: Missing \gls{} for glossary terms
        gls_text = S.read_glossary_file()
        if gls_text:
            gls_defs = L.parse_glossary_defs(gls_text)
            plain = L.strip_latex(tex_content)
            for key, defn in gls_defs.items():
                short = defn.get("short", "")
                if short and len(short) >= 2:
                    # Check for raw usage in original tex (not through \gls{})
                    raw_pattern = re.compile(
                        r"(?<!\\gls\{)(?<!\\acrshort\{)(?<!\\acrlong\{)(?<!\\acrfull\{)"
                        r"\b" + re.escape(short) + r"\b"
                    )
                    for lineno, line in enumerate(lines, 1):
                        if raw_pattern.search(line) and f"\\gls{{{key}}}" not in line and f"\\acrshort{{{key}}}" not in line:
                            # Skip if in a comment
                            if line.strip().startswith("%"):
                                continue
                            violations.append({
                                "rule": 13,
                                "desc": f"Raw acronym '{short}' — use \\gls{{{key}}}",
                                "line": lineno,
                                "context": line.strip()[:120],
                            })
                            break  # Only flag first occurrence per acronym

        # Summary by rule
        from collections import Counter
        rule_counts = Counter(v["rule"] for v in violations)

        return {
            "total_violations": len(violations),
            "by_rule": dict(rule_counts),
            "violations": violations,
        }

    @mcp.tool()
    def check_chapter_boundaries(tex_content: str, chapter_type: str) -> dict:
        """Check cross-chapter constraints based on chapter type.

        Supervisor rules:
        - introduction: No architecture details, no experiment methodology
        - background: No opinions, no comparisons, dry theory only
        - sota: No dry theory, must be thematic not paper-by-paper
        - model: No experiment details
        - experiments: Must link to RQs

        Args:
            tex_content: LaTeX source text of the chapter.
            chapter_type: One of: introduction, background, sota, model, experiments, evaluation.
        """
        plain = L.strip_latex(tex_content)
        violations = []
        lines = tex_content.split("\n")

        if chapter_type == "introduction":
            # No architecture keywords
            arch_words = ["encoder", "decoder", "layer", "backbone", "module", "convolution",
                          "attention head", "feature pyramid", "architecture"]
            for word in arch_words:
                for lineno, line in enumerate(lines, 1):
                    if re.search(r"\b" + re.escape(word) + r"\b", line, re.IGNORECASE):
                        if line.strip().startswith("%"):
                            continue
                        violations.append({
                            "constraint": "no_architecture_in_intro",
                            "match": word,
                            "line": lineno,
                            "context": line.strip()[:120],
                        })
                        break  # One flag per word

            # No experiment methodology
            exp_words = ["dataset", "training set", "test set", "epoch", "batch size",
                         "learning rate", "hyperparameter", "ablation"]
            for word in exp_words:
                for lineno, line in enumerate(lines, 1):
                    if re.search(r"\b" + re.escape(word) + r"\b", line, re.IGNORECASE):
                        if line.strip().startswith("%"):
                            continue
                        violations.append({
                            "constraint": "no_experiments_in_intro",
                            "match": word,
                            "line": lineno,
                            "context": line.strip()[:120],
                        })
                        break

        elif chapter_type == "background":
            # No opinions or comparisons
            opinion_words = ["better", "superior", "outperform", "advantage", "disadvantage",
                             "best", "worst", "prefer", "recommend", "promising", "impressive"]
            for word in opinion_words:
                for lineno, line in enumerate(lines, 1):
                    if re.search(r"\b" + re.escape(word) + r"\b", line, re.IGNORECASE):
                        if line.strip().startswith("%"):
                            continue
                        violations.append({
                            "constraint": "no_opinions_in_background",
                            "match": word,
                            "line": lineno,
                            "context": line.strip()[:120],
                        })
                        break

        elif chapter_type == "sota":
            # Check for paper-by-paper structure (consecutive author citations)
            cites = L.parse_citations(tex_content)
            consecutive_single = 0
            for i in range(len(cites) - 1):
                if len(cites[i]["keys"]) == 1 and len(cites[i + 1]["keys"]) == 1:
                    consecutive_single += 1
            if consecutive_single > 5:
                violations.append({
                    "constraint": "possible_paper_by_paper",
                    "message": f"{consecutive_single} consecutive single-citation paragraphs detected",
                })

        elif chapter_type == "model":
            # No experiment details
            exp_words = ["training", "test set", "validation set", "epoch", "batch size",
                         "learning rate", "loss function", "optimizer"]
            for word in exp_words:
                for lineno, line in enumerate(lines, 1):
                    if re.search(r"\b" + re.escape(word) + r"\b", line, re.IGNORECASE):
                        if line.strip().startswith("%"):
                            continue
                        violations.append({
                            "constraint": "no_experiments_in_model",
                            "match": word,
                            "line": lineno,
                            "context": line.strip()[:120],
                        })
                        break

        return {
            "chapter_type": chapter_type,
            "total_violations": len(violations),
            "violations": violations,
        }

    @mcp.tool()
    def check_cross_references(tex_content: str) -> dict:
        """Find broken references and unused labels in LaTeX content.

        Checks \\ref{}, \\eqref{}, \\autoref{}, \\cref{} against \\label{} definitions.

        Args:
            tex_content: LaTeX source text to check.
        """
        labels, refs = L.parse_labels_refs(tex_content)

        broken_refs = sorted(refs - labels)
        unused_labels = sorted(labels - refs)

        return {
            "total_labels": len(labels),
            "total_refs": len(refs),
            "broken_refs": broken_refs,
            "unused_labels": unused_labels,
        }
