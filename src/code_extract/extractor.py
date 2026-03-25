"""AST-based code extractor for SMCM-MCFNet codebase.

Parses Python source files into structured chunks suitable for
nugget extraction via LLM. Also parses YAML run configs.
"""

import ast
import json
import os
import textwrap
from pathlib import Path
from typing import Optional

import yaml

from src.utils import load_config


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _get_docstring(node: ast.AST) -> str:
    """Extract docstring from a class or function node."""
    return ast.get_docstring(node) or ""


def _format_args(args: ast.arguments) -> str:
    """Format function arguments as a readable signature."""
    parts = []
    defaults_offset = len(args.args) - len(args.defaults)
    for i, arg in enumerate(args.args):
        name = arg.arg
        ann = ast.unparse(arg.annotation) if arg.annotation else None
        default_idx = i - defaults_offset
        default = ast.unparse(args.defaults[default_idx]) if default_idx >= 0 else None
        s = f"{name}: {ann}" if ann else name
        if default:
            s += f" = {default}"
        parts.append(s)
    if args.vararg:
        parts.append(f"*{args.vararg.arg}")
    for kw in args.kwonlyargs:
        name = kw.arg
        ann = ast.unparse(kw.annotation) if kw.annotation else None
        parts.append(f"{name}: {ann}" if ann else name)
    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")
    return ", ".join(parts)


def _get_return_annotation(node: ast.FunctionDef) -> str:
    if node.returns:
        return ast.unparse(node.returns)
    return ""


def _get_base_classes(node: ast.ClassDef) -> list[str]:
    return [ast.unparse(b) for b in node.bases]


def _get_assignments(body: list[ast.stmt]) -> list[dict]:
    """Extract self.xxx = ... assignments from __init__."""
    assignments = []
    for stmt in body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                    if target.value.id == "self":
                        assignments.append({
                            "name": target.attr,
                            "value": ast.unparse(stmt.value)[:120],
                        })
        elif isinstance(stmt, ast.AnnAssign) and stmt.target:
            if isinstance(stmt.target, ast.Attribute) and isinstance(stmt.target.value, ast.Name):
                if stmt.target.value.id == "self":
                    val = ast.unparse(stmt.value)[:120] if stmt.value else ""
                    assignments.append({
                        "name": stmt.target.attr,
                        "value": val,
                    })
    return assignments


def _get_dataclass_fields(node: ast.ClassDef) -> list[dict]:
    """Extract field definitions from a dataclass."""
    fields = []
    for stmt in node.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            field_info = {
                "name": stmt.target.id,
                "type": ast.unparse(stmt.annotation) if stmt.annotation else "",
            }
            if stmt.value:
                field_info["default"] = ast.unparse(stmt.value)[:120]
            # Check for inline comment (not available in AST, skip)
            fields.append(field_info)
    return fields


def _is_dataclass(node: ast.ClassDef) -> bool:
    """Check if class has @dataclass decorator."""
    for dec in node.decorator_list:
        name = ""
        if isinstance(dec, ast.Name):
            name = dec.id
        elif isinstance(dec, ast.Attribute):
            name = dec.attr
        elif isinstance(dec, ast.Call):
            if isinstance(dec.func, ast.Name):
                name = dec.func.id
            elif isinstance(dec.func, ast.Attribute):
                name = dec.func.attr
        if name == "dataclass":
            return True
    return False


# ---------------------------------------------------------------------------
# File-level extraction
# ---------------------------------------------------------------------------

def extract_file(filepath: str, code_root: str) -> Optional[dict]:
    """Extract structured information from a single Python file.

    Returns a dict with module info, classes, functions, and a formatted
    text chunk ready for nugget extraction.
    """
    try:
        with open(filepath) as f:
            source = f.read()
        tree = ast.parse(source, filename=filepath)
    except (SyntaxError, UnicodeDecodeError):
        return None

    rel_path = os.path.relpath(filepath, os.path.dirname(code_root))
    module_name = rel_path.replace("/", ".").replace(".py", "")
    module_doc = ast.get_docstring(tree) or ""

    classes = []
    functions = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            cls_info = _extract_class(node)
            classes.append(cls_info)
        elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            if not node.name.startswith("_") or node.name in ("__init__", "__call__"):
                func_info = _extract_function(node)
                functions.append(func_info)

    if not classes and not functions and not module_doc:
        return None

    # Build formatted chunk text
    chunk_text = _format_chunk(module_name, rel_path, module_doc, classes, functions)

    return {
        "file": rel_path,
        "module": module_name,
        "module_doc": module_doc,
        "classes": classes,
        "functions": functions,
        "chunk_text": chunk_text,
    }


def _extract_class(node: ast.ClassDef) -> dict:
    bases = _get_base_classes(node)
    docstring = _get_docstring(node)
    is_dc = _is_dataclass(node)

    methods = []
    init_components = []
    dc_fields = []

    if is_dc:
        dc_fields = _get_dataclass_fields(node)

    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if item.name == "__init__":
                init_components = _get_assignments(item.body)
            # Include key methods
            if item.name in ("forward", "__init__", "__call__", "training_step",
                             "validation_step", "configure_optimizers", "build",
                             "loss", "compute_loss", "predict", "encode", "decode"):
                methods.append(_extract_function(item))
            elif not item.name.startswith("_"):
                methods.append(_extract_function(item))

    return {
        "name": node.name,
        "bases": bases,
        "docstring": docstring,
        "is_dataclass": is_dc,
        "dataclass_fields": dc_fields,
        "methods": methods,
        "init_components": init_components,
    }


def _extract_function(node) -> dict:
    sig = _format_args(node.args)
    ret = _get_return_annotation(node)
    docstring = _get_docstring(node)
    return {
        "name": node.name,
        "signature": sig,
        "return_type": ret,
        "docstring": docstring,
    }


# ---------------------------------------------------------------------------
# Chunk formatting
# ---------------------------------------------------------------------------

def _format_chunk(module: str, rel_path: str, module_doc: str,
                  classes: list[dict], functions: list[dict]) -> str:
    """Format extracted info into a structured text chunk for LLM ingestion."""
    lines = [
        f"[Module: {module}]",
        f"[File: {rel_path}]",
    ]
    if module_doc:
        lines.append(f"Module purpose: {module_doc.split(chr(10))[0]}")
    lines.append("")

    for cls in classes:
        bases_str = ", ".join(cls["bases"]) if cls["bases"] else "object"
        lines.append(f"Class: {cls['name']}({bases_str})")
        if cls["docstring"]:
            lines.append(f"  Purpose: {cls['docstring'].split(chr(10))[0]}")

        if cls["is_dataclass"] and cls["dataclass_fields"]:
            lines.append("  Dataclass fields:")
            for f in cls["dataclass_fields"]:
                default = f" = {f['default']}" if "default" in f else ""
                lines.append(f"    {f['name']}: {f['type']}{default}")

        if cls["init_components"]:
            lines.append("  Components (from __init__):")
            for comp in cls["init_components"][:15]:  # cap at 15 to avoid noise
                lines.append(f"    self.{comp['name']} = {comp['value']}")

        for method in cls["methods"]:
            ret = f" -> {method['return_type']}" if method["return_type"] else ""
            lines.append(f"  Method: {method['name']}({method['signature']}){ret}")
            if method["docstring"]:
                first_line = method["docstring"].split("\n")[0]
                lines.append(f"    {first_line}")

        lines.append("")

    for func in functions:
        ret = f" -> {func['return_type']}" if func["return_type"] else ""
        lines.append(f"Function: {func['name']}({func['signature']}){ret}")
        if func["docstring"]:
            lines.append(f"  {func['docstring'].split(chr(10))[0]}")
        lines.append("")

    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# YAML config extraction
# ---------------------------------------------------------------------------

def extract_yaml_config(filepath: str, config_root: str) -> Optional[dict]:
    """Extract a YAML run config as a structured chunk."""
    try:
        with open(filepath) as f:
            data = yaml.safe_load(f)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    rel_path = os.path.relpath(filepath, os.path.dirname(config_root))
    run_name = Path(filepath).stem

    # Determine experiment category from directory structure
    parts = Path(filepath).relative_to(config_root).parts
    category = parts[0] if len(parts) > 1 else "general"

    lines = [
        f"[Config: {run_name}]",
        f"[File: {rel_path}]",
        f"[Category: {category}]",
        "",
    ]

    def _flatten(d, prefix=""):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                _flatten(v, key)
            else:
                lines.append(f"  {key}: {v}")

    lines.append("Parameters:")
    _flatten(data)

    chunk_text = "\n".join(lines).strip()
    return {
        "file": rel_path,
        "run_name": run_name,
        "category": category,
        "config": data,
        "chunk_text": chunk_text,
    }


# ---------------------------------------------------------------------------
# Directory-level extraction
# ---------------------------------------------------------------------------

SKIP_DIRS = {"__pycache__", ".git", "test", "tests", "eval"}
SKIP_FILES = {"__init__.py"}


def extract_codebase(code_dir: str, output_dir: str):
    """Extract all Python files from the codebase into structured chunks."""
    code_dir = os.path.expanduser(code_dir)
    os.makedirs(output_dir, exist_ok=True)

    chunks = []
    for root, dirs, files in os.walk(code_dir):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in sorted(files):
            if not fname.endswith(".py") or fname in SKIP_FILES:
                continue
            filepath = os.path.join(root, fname)
            result = extract_file(filepath, code_dir)
            if result:
                chunks.append(result)
                print(f"  Extracted: {result['file']}")

    out_path = os.path.join(output_dir, "code_chunks.json")
    with open(out_path, "w") as f:
        json.dump(chunks, f, indent=2)
    print(f"\n  {len(chunks)} code chunks -> {out_path}")
    return chunks


def extract_configs(config_dir: str, output_dir: str):
    """Extract all YAML run configs into structured chunks."""
    config_dir = os.path.expanduser(config_dir)
    os.makedirs(output_dir, exist_ok=True)

    chunks = []
    for root, dirs, files in os.walk(config_dir):
        for fname in sorted(files):
            if not fname.endswith(".yaml") and not fname.endswith(".yml"):
                continue
            filepath = os.path.join(root, fname)
            result = extract_yaml_config(filepath, config_dir)
            if result:
                chunks.append(result)
                print(f"  Extracted config: {result['run_name']}")

    out_path = os.path.join(output_dir, "config_chunks.json")
    with open(out_path, "w") as f:
        json.dump(chunks, f, indent=2)
    print(f"\n  {len(chunks)} config chunks -> {out_path}")
    return chunks


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Extract code chunks from SMCM-MCFNet")
    ap.add_argument("-c", "--config", default="config.yaml", help="Config file path")
    args = ap.parse_args()

    cfg = load_config(args.config)
    code_dir = cfg["paths"]["code_dir"]
    config_dir = cfg["paths"].get("code_config_dir", "")
    output_dir = cfg["paths"]["code_chunk_dir"]

    print("[code_extract] Extracting Python source files...")
    code_chunks = extract_codebase(code_dir, output_dir)

    if config_dir:
        print("\n[code_extract] Extracting YAML run configs...")
        config_chunks = extract_configs(config_dir, output_dir)

    print("\n[code_extract] Done.")


if __name__ == "__main__":
    main()
