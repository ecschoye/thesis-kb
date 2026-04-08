"""Writing quality and structure tools for thesis-kb."""

from __future__ import annotations

import re
from statistics import mean, stdev

from . import _shared as S
from . import _latex as L
from . import _text as T


def register(mcp):

    @mcp.tool()
    def writing_stats(tex_content: str) -> dict:
        """Word counts, readability scores, and structural metrics for LaTeX content.

        Strips LaTeX commands and computes Flesch-Kincaid grade level,
        per-section word counts, and structural element counts.

        Args:
            tex_content: LaTeX source text.
        """
        plain = L.strip_latex(tex_content)
        fk = T.flesch_kincaid(plain)
        n_paragraphs = T.count_paragraphs(plain)

        # Count structural elements
        n_figures = len(L.parse_figures(tex_content))
        n_equations = len(re.findall(r"\\begin\{equation\}", tex_content))
        n_citations = len(L.parse_citations(tex_content))

        # Per-section breakdown
        sections = L.parse_sections(tex_content)
        section_stats = []
        for sec in sections:
            sec_plain = L.strip_latex(sec["content"])
            section_stats.append({
                "title": sec["title"],
                "level": sec["level"],
                "words": T.word_count(sec_plain),
            })

        return {
            **fk,
            "paragraphs": n_paragraphs,
            "figures": n_figures,
            "equations": n_equations,
            "citations": n_citations,
            "sections": section_stats,
        }

    @mcp.tool()
    def chapter_overview(tex_content: str) -> dict:
        """Structural overview: section hierarchy with word, citation, and figure counts.

        Args:
            tex_content: LaTeX source text.
        """
        sections = L.parse_sections(tex_content)
        if not sections:
            plain = L.strip_latex(tex_content)
            return {
                "sections": [],
                "total_words": T.word_count(plain),
                "total_citations": len(L.parse_citations(tex_content)),
                "total_figures": len(L.parse_figures(tex_content)),
            }

        results = []
        total_words = 0
        total_cites = 0
        total_figs = 0

        for sec in sections:
            sec_plain = L.strip_latex(sec["content"])
            words = T.word_count(sec_plain)
            cites = len(L.parse_citations(sec["content"]))
            figs = len(L.parse_figures(sec["content"]))
            total_words += words
            total_cites += cites
            total_figs += figs

            results.append({
                "title": sec["title"],
                "level": sec["level"],
                "line": sec["line"],
                "words": words,
                "citations": cites,
                "figures": figs,
            })

        return {
            "sections": results,
            "total_words": total_words,
            "total_citations": total_cites,
            "total_figures": total_figs,
        }

    @mcp.tool()
    def check_terminology(tex_content: str) -> dict:
        """Check for inconsistent terminology and glossary compliance.

        Detects: terms used without \\gls{}, alternating synonym usage,
        and glossary keys used but not defined.

        Args:
            tex_content: LaTeX source text.
        """
        plain = L.strip_latex(tex_content)
        gls_text = S.read_glossary_file()
        gls_defs = L.parse_glossary_defs(gls_text) if gls_text else {}
        gls_used = set(L.parse_gls_usage(tex_content))

        issues = []

        # Check for full-form usage of glossary acronyms without \gls{}
        for key, defn in gls_defs.items():
            short = defn.get("short", "")
            long_form = defn.get("long", "")
            if short and key not in gls_used:
                # Check if the short form appears raw in text
                if re.search(r"\b" + re.escape(short) + r"\b", plain):
                    issues.append({
                        "type": "raw_acronym",
                        "key": key,
                        "short": short,
                        "long": long_form,
                        "message": f"'{short}' used without \\gls{{{key}}}",
                    })

        # Domain-specific synonym groups
        synonyms = [
            ("event camera", "DVS", "dynamic vision sensor", "neuromorphic camera"),
            ("spiking neural network", "SNN"),
            ("convolutional neural network", "CNN"),
            ("object detection", "object recognition"),
            ("membrane potential", "membrane voltage"),
            ("leaky integrate-and-fire", "LIF"),
            ("bounding box", "bbox"),
        ]

        for group in synonyms:
            found = []
            for term in group:
                count = len(re.findall(re.escape(term), plain, re.IGNORECASE))
                if count > 0:
                    found.append({"term": term, "count": count})
            if len(found) > 1:
                issues.append({
                    "type": "synonym_alternation",
                    "terms": found,
                    "message": f"Multiple synonyms used: {', '.join(f['term'] for f in found)}",
                })

        return {
            "glossary_keys_defined": len(gls_defs),
            "glossary_keys_used": len(gls_used),
            "issues": issues,
        }

    @mcp.tool()
    def missing_definitions(tex_content: str) -> dict:
        """Find undefined acronyms and terms missing \\gls{} introduction.

        Scans for uppercase sequences (2+ chars) not defined in glossary
        or introduced via \\acrfull{}.

        Args:
            tex_content: LaTeX source text.
        """
        plain = L.strip_latex(tex_content)
        gls_text = S.read_glossary_file()
        gls_defs = L.parse_glossary_defs(gls_text) if gls_text else {}

        # Find all potential acronyms (2+ uppercase letters)
        acronyms = set(re.findall(r"\b[A-Z]{2,}\b", plain))

        # Common acronyms to skip
        skip = {
            "IEEE", "ACM", "CVPR", "ICCV", "ECCV", "AAAI", "NIPS", "ICML",
            "FPGA", "GPU", "CPU", "RAM", "RGB", "USB", "API", "PDF", "URL",
            "MATH", "TODO", "NOTE", "TABLE", "FIGURE",
        }

        # Check which are defined in glossary
        gls_shorts = {d.get("short", "").upper() for d in gls_defs.values() if d.get("short")}

        # Check for \acrfull{} usage in the original text
        acrfull_used = set(re.findall(r"\\acrfull\{(\w+)\}", tex_content))
        acrfull_shorts = {gls_defs.get(k, {}).get("short", "").upper() for k in acrfull_used}

        undefined = []
        for acr in sorted(acronyms):
            if acr in skip:
                continue
            if acr in gls_shorts or acr in acrfull_shorts:
                continue
            # Check if it appears in glossary keys (case-insensitive)
            if acr.lower() in gls_defs:
                continue
            undefined.append(acr)

        # Check \gls{} keys not in glossary
        gls_used = set(L.parse_gls_usage(tex_content))
        gls_undefined = sorted(gls_used - set(gls_defs.keys()))

        return {
            "undefined_acronyms": undefined,
            "gls_keys_not_in_glossary": gls_undefined,
            "total_acronyms_found": len(acronyms),
        }

    @mcp.tool()
    def section_balance(tex_content: str) -> dict:
        """Assess whether sections are proportionally balanced in length.

        Computes word counts per top-level section, coefficient of variation,
        and flags sections that are <25% or >300% of mean length.

        Args:
            tex_content: LaTeX source text.
        """
        sections = L.parse_sections(tex_content)
        # Only consider top-level sections (level 1 = \\section)
        top_sections = [s for s in sections if s["level"] <= 1]
        if len(top_sections) < 2:
            return {"sections": [], "message": "Need at least 2 sections to assess balance"}

        counts = []
        details = []
        for sec in top_sections:
            plain = L.strip_latex(sec["content"])
            wc = T.word_count(plain)
            counts.append(wc)
            details.append({"title": sec["title"], "words": wc})

        avg = mean(counts)
        sd = stdev(counts) if len(counts) > 1 else 0
        cv = round(sd / avg, 2) if avg > 0 else 0

        for d in details:
            ratio = d["words"] / avg if avg > 0 else 0
            d["ratio_to_mean"] = round(ratio, 2)
            d["flagged"] = ratio < 0.25 or ratio > 3.0

        flagged = [d for d in details if d["flagged"]]
        return {
            "sections": details,
            "mean_words": round(avg),
            "std_dev": round(sd),
            "coefficient_of_variation": cv,
            "flagged_sections": flagged,
        }

    @mcp.tool()
    def check_figure_discussion(tex_content: str) -> dict:
        """Verify that every figure and table is referenced in the text body.

        Flags figures/tables with labels never \\ref'd, and \\ref's to
        non-existent figure/table labels.

        Args:
            tex_content: LaTeX source text.
        """
        figures = L.parse_figures(tex_content)
        labels, refs = L.parse_labels_refs(tex_content)

        # Collect figure/table labels
        fig_labels = {f["label"] for f in figures if f["label"]}

        not_referenced = []
        for fig in figures:
            if fig["label"] and fig["label"] not in refs:
                not_referenced.append({
                    "env_type": fig["env_type"],
                    "label": fig["label"],
                    "caption": (fig["caption"][:80] + "...") if fig.get("caption") and len(fig["caption"]) > 80 else fig.get("caption"),
                    "line": fig["line"],
                })

        # Find refs to fig/tab labels that don't exist
        fig_ref_pattern = re.compile(r"fig:|tab:|table:|figure:", re.IGNORECASE)
        broken_refs = []
        for ref in refs:
            if fig_ref_pattern.search(ref) and ref not in labels:
                broken_refs.append(ref)

        return {
            "total_figures": sum(1 for f in figures if f["env_type"] == "figure"),
            "total_tables": sum(1 for f in figures if f["env_type"] == "table"),
            "not_referenced": not_referenced,
            "broken_refs": broken_refs,
        }
