"""The MCP server registers the expected tools (introspected without spawning a process)."""

from __future__ import annotations

import asyncio

from agentic_workbench.mcp_server import server


def test_server_exposes_all_tools() -> None:
    tools = asyncio.run(server.list_tools())
    names = {t.name for t in tools}
    assert {"calculator", "knowledge_search", "word_count"} <= names


def test_server_tools_have_schemas() -> None:
    tools = asyncio.run(server.list_tools())
    # Each MCP tool advertises an input schema so clients know how to call it.
    assert all(t.inputSchema for t in tools)
