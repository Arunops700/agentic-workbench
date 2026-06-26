"""agentic-workbench: a tool-using ReAct agent in two implementations.

The same `Policy` + `ToolRegistry` drive a from-scratch ReAct loop (`react.py`, for learning the
mechanics) and a production-shaped LangGraph state machine (`graph.py`, with memory). The tools are
also exposed over the Model Context Protocol (`mcp_server.py`), so any MCP client can use them.
"""

from agentic_workbench.models import AgentResult, Decision, ToolCall, ToolResult

__version__ = "0.1.0"

__all__ = ["AgentResult", "Decision", "ToolCall", "ToolResult", "__version__"]
