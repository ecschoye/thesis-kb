"""LaTeX and BibTeX parsing utilities for thesis-kb tools.

All parsing uses stdlib only (re, difflib) — no external dependencies.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher


# ---------------------------------------------------------------------------
# Citation extraction
# ---------------------------------------------------------------------------

_CITE_RE = re.compile(
    r"\\(cite[pt]?|citeauthor|citeyear)\{([^}]+)\}"
)


def parse_citations(tex: str) -> list[dict]:
    """Extract all citation commands with context.

    Returns list of dicts:
        {cmd, keys: [str], line: int, context: str}
    """
    lines = tex.split("\n")
    results = []
    for lineno, line in enumerate(lines, 1):
        for m in _CITE_RE.finditer(line):
            cmd = m.group(1)
            raw_keys = m.group(2)
            keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
            results.append({
                "cmd": cmd,
                "keys": keys,
                "line": lineno,
                "context": line.strip(),
            })
    return results


def extract_all_cite_keys(tex: str) -> list[str]:
    """Return a flat deduplicated list of all cite keys in order of appearance."""
    seen = set()
    keys = []
    for cit in parse_citations(tex):
        for k in cit["keys"]:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    return keys


# ---------------------------------------------------------------------------
# BibTeX parsing (state-machine, handles nested braces)
# ---------------------------------------------------------------------------

def parse_bib_entries(bib_text: str) -> list[dict]:
    """Parse a .bib file into structured entries.

    Returns list of dicts:
        {key, entry_type, fields: {field_name: value}, line: int}
    """
    entries = []
    # Match @type{key, ... }
    entry_start = re.compile(r"@(\w+)\s*\{(.+)")
    i = 0
    lines = bib_text.split("\n")

    while i < len(lines):
        m = entry_start.match(lines[i].strip())
        if not m:
            i += 1
            continue

        entry_type = m.group(1).lower()
        if entry_type in ("comment", "preamble", "string"):
            i += 1
            continue

        rest = m.group(2)
        # Key is everything before the first comma
        if "," in rest:
            key = rest.split(",", 1)[0].strip()
        else:
            key = rest.strip().rstrip("}")

        # Collect the full entry body by tracking brace depth
        start_line = i + 1
        brace_depth = 1
        body_lines = [rest.split(",", 1)[1] if "," in rest else ""]
        i += 1
        while i < len(lines) and brace_depth > 0:
            line = lines[i]
            for ch in line:
                if ch == "{":
                    brace_depth += 1
                elif ch == "}":
                    brace_depth -= 1
            if brace_depth > 0:
                body_lines.append(line)
            else:
                # Last line — take content before the closing brace
                idx = line.rfind("}")
                body_lines.append(line[:idx])
            i += 1

        body = "\n".join(body_lines)
        fields = _parse_bib_fields(body)
        entries.append({
            "key": key,
            "entry_type": entry_type,
            "fields": fields,
            "line": start_line,
        })

    return entries


def _parse_bib_fields(body: str) -> dict[str, str]:
    """Parse field = {value} or field = "value" pairs from a bib entry body."""
    fields: dict[str, str] = {}
    # Match field_name = {value} or field_name = "value" or field_name = number
    field_re = re.compile(r"(\w+)\s*=\s*")
    pos = 0
    while pos < len(body):
        m = field_re.search(body, pos)
        if not m:
            break
        name = m.group(1).lower()
        val_start = m.end()
        # Skip whitespace
        while val_start < len(body) and body[val_start] in " \t\n":
            val_start += 1
        if val_start >= len(body):
            break

        if body[val_start] == "{":
            value, end = _extract_braced(body, val_start)
        elif body[val_start] == '"':
            end = body.index('"', val_start + 1) + 1
            value = body[val_start + 1 : end - 1]
        else:
            # Bare number or string reference
            end = val_start
            while end < len(body) and body[end] not in ",}\n":
                end += 1
            value = body[val_start:end].strip()

        fields[name] = value.strip()
        pos = end + 1

    return fields


def _extract_braced(text: str, start: int) -> tuple[str, int]:
    """Extract content between balanced braces starting at position start."""
    depth = 0
    i = start
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i], i + 1
        i += 1
    return text[start + 1 :], len(text)


# ---------------------------------------------------------------------------
# Section parsing
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(
    r"^\\(chapter|section|subsection|subsubsection)\*?\{(.+?)\}",
    re.MULTILINE,
)


def parse_sections(tex: str) -> list[dict]:
    """Parse section hierarchy from LaTeX.

    Returns list of dicts:
        {level, title, line, start_pos, content}
    where content is the text between this heading and the next.
    """
    levels = {"chapter": 0, "section": 1, "subsection": 2, "subsubsection": 3}
    matches = list(_SECTION_RE.finditer(tex))
    if not matches:
        return []

    sections = []
    lines = tex[:matches[0].start()].count("\n")
    for idx, m in enumerate(matches):
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(tex)
        line = tex[: m.start()].count("\n") + 1
        sections.append({
            "level": levels.get(m.group(1), 1),
            "cmd": m.group(1),
            "title": m.group(2).strip(),
            "line": line,
            "content": tex[start:end].strip(),
        })
    return sections


# ---------------------------------------------------------------------------
# Labels and references
# ---------------------------------------------------------------------------

_LABEL_RE = re.compile(r"\\label\{([^}]+)\}")
_REF_RE = re.compile(r"\\(?:eq)?(?:ref|autoref|cref|Cref)\{([^}]+)\}")


def parse_labels_refs(tex: str) -> tuple[set[str], set[str]]:
    """Return (labels_defined, refs_used) from LaTeX source."""
    labels = set(_LABEL_RE.findall(tex))
    refs = set(_REF_RE.findall(tex))
    return labels, refs


# ---------------------------------------------------------------------------
# Figures and tables
# ---------------------------------------------------------------------------

_ENV_RE = re.compile(
    r"\\begin\{(figure|table)\*?\}(.*?)\\end\{\1\*?\}",
    re.DOTALL,
)


def parse_figures(tex: str) -> list[dict]:
    """Extract figure and table environments.

    Returns list of dicts:
        {env_type, label, caption, line}
    """
    results = []
    for m in _ENV_RE.finditer(tex):
        env_type = m.group(1)
        body = m.group(2)
        line = tex[: m.start()].count("\n") + 1

        label_m = _LABEL_RE.search(body)
        label = label_m.group(1) if label_m else None

        cap_m = re.search(r"\\caption\{(.+?)\}", body, re.DOTALL)
        caption = cap_m.group(1).strip() if cap_m else None

        results.append({
            "env_type": env_type,
            "label": label,
            "caption": caption,
            "line": line,
        })
    return results


# ---------------------------------------------------------------------------
# Glossary
# ---------------------------------------------------------------------------

_NEWACRONYM_RE = re.compile(
    r"\\newacronym(?:\[.*?\])?\{(\w+)\}\{([^}]+)\}\{([^}]+)\}"
)
_NEWGLOSSARYENTRY_RE = re.compile(
    r"\\newglossaryentry\{(\w+)\}\{[^}]*name\s*=\s*\{?([^},]+)"
)
_GLS_RE = re.compile(r"\\(?:gls|acrshort|acrlong|acrfull|Gls|Acrshort|Acrlong|Acrfull)\{(\w+)\}")


def parse_glossary_defs(glossary_tex: str) -> dict[str, dict]:
    """Parse \\newacronym and \\newglossaryentry definitions.

    Returns {key: {short, long}} for acronyms, {key: {name}} for glossary entries.
    """
    defs: dict[str, dict] = {}
    for m in _NEWACRONYM_RE.finditer(glossary_tex):
        defs[m.group(1)] = {"short": m.group(2), "long": m.group(3)}
    for m in _NEWGLOSSARYENTRY_RE.finditer(glossary_tex):
        if m.group(1) not in defs:
            defs[m.group(1)] = {"name": m.group(2).strip()}
    return defs


def parse_gls_usage(tex: str) -> list[str]:
    """Return all glossary keys used via \\gls{}, \\acrshort{}, etc."""
    return _GLS_RE.findall(tex)


# ---------------------------------------------------------------------------
# LaTeX stripping
# ---------------------------------------------------------------------------

def strip_latex(tex: str) -> str:
    """Strip LaTeX commands to produce plain text for analysis.

    Removes comments, commands, environments, math, and normalizes whitespace.
    """
    # Remove comments (but not escaped %)
    text = re.sub(r"(?<!\\)%.*$", "", tex, flags=re.MULTILINE)
    # Remove \begin{...} and \end{...}
    text = re.sub(r"\\(?:begin|end)\{[^}]+\}", " ", text)
    # Remove inline math $...$
    text = re.sub(r"\$[^$]+\$", " MATH ", text)
    # Remove display math \[...\] and $$...$$
    text = re.sub(r"\\\[.*?\\\]", " MATH ", text, flags=re.DOTALL)
    text = re.sub(r"\$\$.*?\$\$", " MATH ", text, flags=re.DOTALL)
    # Remove \command{content} -> keep content
    text = re.sub(r"\\(?:textbf|textit|emph|underline|texttt|text)\{([^}]*)\}", r"\1", text)
    # Remove cite/ref commands entirely
    text = re.sub(r"\\(?:cite[pt]?|citeauthor|citeyear|ref|eqref|autoref|label)\{[^}]*\}", "", text)
    # Remove \gls-family but keep the key as text
    text = re.sub(r"\\(?:gls|acrshort|acrlong|acrfull|Gls|Acrshort|Acrlong|Acrfull)\{(\w+)\}", r"\1", text)
    # Remove remaining commands with optional args
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])*(?:\{[^}]*\})*", " ", text)
    # Remove braces
    text = re.sub(r"[{}]", "", text)
    # Remove backslash escapes
    text = re.sub(r"\\.", "", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Fuzzy title matching
# ---------------------------------------------------------------------------

def fuzzy_match_title(
    query: str, candidates: list[str], threshold: float = 0.6
) -> list[tuple[str, float]]:
    """Fuzzy-match a query title against candidates using SequenceMatcher.

    Returns sorted list of (candidate, score) above threshold.
    """
    query_lower = query.lower()
    results = []
    for c in candidates:
        score = SequenceMatcher(None, query_lower, c.lower()).ratio()
        if score >= threshold:
            results.append((c, score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results
