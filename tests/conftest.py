"""Test helpers: the standard registry and a builder for scripted tool calls."""

from __future__ import annotations

import pytest

from agentic_workbench.models import Decision, ToolCall
from agentic_workbench.tools import ToolRegistry, default_registry


@pytest.fixture
def registry() -> ToolRegistry:
    return default_registry()


def calc(expr: str, cid: str = "c0") -> Decision:
    return Decision(tool_calls=[ToolCall(id=cid, name="calculator", args={"expression": expr})])


def finish(text: str) -> Decision:
    return Decision(finish=text)
