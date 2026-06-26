"""The shared tool layer.

Tools are the agent's hands. A `Tool` couples a name + description + JSON-schema parameters (what
the model sees) to a Python callable (what actually runs). The `ToolRegistry` executes them safely —
turning exceptions into error results the model can recover from, never crashing the loop. The same
registry feeds both agent implementations *and* the MCP server, so there's one source of truth.
"""

from __future__ import annotations

import ast
import operator
import re
from collections.abc import Callable
from dataclasses import dataclass

from agentic_workbench.models import ToolCall, ToolResult

# ---------------------------------------------------------------------------
# Tool definition + registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Tool:
    name: str
    description: str
    parameters: dict  # JSON schema for the arguments object
    func: Callable[..., str]


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def execute(self, call: ToolCall) -> ToolResult:
        """Run a tool. Errors become `is_error` results, not exceptions — the agent must recover."""
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(
                id=call.id, name=call.name, output=f"Unknown tool '{call.name}'.", is_error=True
            )
        try:
            output = tool.func(**call.args)
            return ToolResult(id=call.id, name=call.name, output=str(output))
        except Exception as exc:  # surfaced to the model so it can try again
            return ToolResult(id=call.id, name=call.name, output=f"Error: {exc}", is_error=True)

    def to_anthropic_tools(self) -> list[dict]:
        """Render tools in the Anthropic tool-use schema."""
        return [
            {"name": t.name, "description": t.description, "input_schema": t.parameters}
            for t in self._tools.values()
        ]


# ---------------------------------------------------------------------------
# Built-in tools
# ---------------------------------------------------------------------------

_BIN_OPS: dict[type, Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}


def _safe_eval(node: ast.AST) -> float:
    """Evaluate a numeric AST. Only arithmetic — never `eval()` on model output."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_safe_eval(node.operand)
    raise ValueError("Unsupported expression — arithmetic only.")


def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '2 * (3 + 4)'."""
    tree = ast.parse(expression, mode="eval")
    return str(_safe_eval(tree.body))


# A tiny built-in knowledge base. In production this tool would call the deployed RAG service
# (see the rag-knowledge-assistant project); kept self-contained here.
_KB: dict[str, str] = {
    "react": "ReAct interleaves reasoning and acting: the agent thinks, calls a tool, observes the "
    "result, and repeats — which fixes chain-of-thought's habit of hallucinating facts it should "
    "look up.",
    "mcp": "The Model Context Protocol is an open standard for how AI apps expose tools, "
    "resources, and prompts to models. A server publishes capabilities; any client can use them.",
    "langgraph": "LangGraph models an agent as a state machine: nodes do work, edges (including "
    "conditional ones) route control, and a checkpointer persists state across turns.",
}


def knowledge_search(query: str) -> str:
    """Search the built-in knowledge base for a short, relevant snippet."""
    terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    best_key, best_score = None, 0
    for key, text in _KB.items():
        score = len(terms & set(re.findall(r"[a-z0-9]+", (key + " " + text).lower())))
        if score > best_score:
            best_key, best_score = key, score
    if best_key is None:
        return "No relevant information found."
    return _KB[best_key]


def word_count(text: str) -> str:
    """Count the words in a piece of text."""
    return str(len(text.split()))


def default_registry() -> ToolRegistry:
    """The standard tool set, shared by the agents and the MCP server."""
    return ToolRegistry(
        [
            Tool(
                name="calculator",
                description="Evaluate a basic arithmetic expression like '2 * (3 + 4)'.",
                parameters={
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
                func=calculator,
            ),
            Tool(
                name="knowledge_search",
                description="Look up a short fact about ReAct, MCP, or LangGraph.",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                func=knowledge_search,
            ),
            Tool(
                name="word_count",
                description="Count the number of words in some text.",
                parameters={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
                func=word_count,
            ),
        ]
    )
