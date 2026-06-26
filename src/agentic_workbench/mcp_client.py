"""A minimal MCP client that talks to our server over stdio.

Demonstrates the other side of MCP: spawn the server, initialize a session, discover its tools, and
call one — exactly what any MCP-compatible host does. Run with `agent mcp-demo`. Imports are kept
inside the function so importing this module never requires a live server.
"""

from __future__ import annotations


async def demo() -> None:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command="python", args=["-m", "agentic_workbench.mcp_server"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("Tools exposed by the server:")
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description}")

            result = await session.call_tool("calculator", {"expression": "2 * (3 + 4)"})
            first = result.content[0] if result.content else None
            text = getattr(first, "text", "(non-text result)") if first else "(no content)"
            print(f"\ncalculator('2 * (3 + 4)') -> {text}")


def main() -> None:
    import asyncio

    asyncio.run(demo())


if __name__ == "__main__":
    main()
