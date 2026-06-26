"""An MCP server exposing the agent's tools over the Model Context Protocol.

MCP is the emerging standard for how AI apps publish tools/resources to *any* model or client. The
same tool functions the in-process agent uses (`tools.py`) are registered here on a `FastMCP`
server, so a Claude Desktop, an IDE, or our own `mcp_client.py` can call them over stdio — no glue
code per consumer. Run it with `python -m agentic_workbench.mcp_server` (or `agent mcp-serve`).
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from agentic_workbench.tools import calculator, knowledge_search, word_count

# FastMCP infers each tool's JSON schema from the function's type hints + docstring.
server = FastMCP("agentic-workbench")
server.tool()(calculator)
server.tool()(knowledge_search)
server.tool()(word_count)


def main() -> None:
    """Run the server over stdio (the default MCP transport for local tools)."""
    server.run()


if __name__ == "__main__":
    main()
