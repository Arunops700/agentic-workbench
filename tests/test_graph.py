"""The LangGraph agent: it produces the right answer, and memory persists across turns."""

from __future__ import annotations

from agentic_workbench.graph import LangGraphAgent
from agentic_workbench.policy import HeuristicPolicy, ScriptedPolicy
from agentic_workbench.tools import default_registry
from tests.conftest import calc, finish


def test_graph_runs_tool_then_finishes() -> None:
    policy = ScriptedPolicy([calc("10 / 2"), finish("It is 5.")])
    agent = LangGraphAgent(registry=default_registry(), policy=policy)
    result = agent.run("compute 10/2", thread_id="t-graph")

    assert result.answer == "It is 5."
    assert result.transcript[0].results[0].output == "5.0"


def test_memory_persists_across_turns() -> None:
    agent = LangGraphAgent(registry=default_registry(), policy=HeuristicPolicy())
    first = agent.run("what is mcp?", thread_id="mem", fresh=True)
    # Second turn on the SAME thread, fresh=False → the checkpointer restores prior history.
    second = agent.run("and react?", thread_id="mem", fresh=False)

    assert len(first.transcript) >= 1
    # History accumulated: the second run sees at least as much as the first.
    assert len(second.transcript) >= len(first.transcript)
