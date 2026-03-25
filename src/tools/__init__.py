"""Tool modules for the thesis-kb MCP server."""

from . import citations, quality, gaps, conflicts, formatting, language, code


def register_all(mcp):
    """Register all tool modules with the MCP server."""
    citations.register(mcp)
    quality.register(mcp)
    gaps.register(mcp)
    conflicts.register(mcp)
    formatting.register(mcp)
    language.register(mcp)
    code.register(mcp)
