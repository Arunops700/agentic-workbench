"""Tools must compute correctly and fail safely (errors become results, not crashes)."""

from __future__ import annotations

from agentic_workbench.models import ToolCall
from agentic_workbench.tools import ToolRegistry, calculator, default_registry, knowledge_search


def test_calculator_basic() -> None:
    assert calculator("2 * (3 + 4)") == "14.0"


def test_calculator_rejects_non_arithmetic() -> None:
    registry = default_registry()
    result = registry.execute(
        ToolCall(id="x", name="calculator", args={"expression": "__import__('os')"})
    )
    assert result.is_error  # safely returned, not raised


def test_knowledge_search_finds_topic() -> None:
    assert "Model Context Protocol" in knowledge_search("what is mcp")


def test_unknown_tool_is_error_not_exception() -> None:
    registry = ToolRegistry()
    result = registry.execute(ToolCall(id="x", name="nope", args={}))
    assert result.is_error
    assert "Unknown tool" in result.output


def test_anthropic_tool_schema_shape() -> None:
    tools = default_registry().to_anthropic_tools()
    assert {t["name"] for t in tools} == {"calculator", "knowledge_search", "word_count"}
    assert all("input_schema" in t for t in tools)
