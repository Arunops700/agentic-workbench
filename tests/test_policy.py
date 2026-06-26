"""Policies: scripted sequencing and the offline heuristic's routing."""

from __future__ import annotations

from agentic_workbench.models import Step
from agentic_workbench.policy import HeuristicPolicy, ScriptedPolicy
from tests.conftest import calc, finish


def test_scripted_policy_returns_in_order_then_finishes() -> None:
    policy = ScriptedPolicy([calc("1+1"), finish("done")])
    assert not policy.decide("t", []).is_final  # first: tool call
    assert policy.decide("t", [Step()]).finish == "done"  # second: finish
    assert policy.decide("t", [Step(), Step()]).is_final  # exhausted → finishes


def test_heuristic_routes_math_to_calculator() -> None:
    decision = HeuristicPolicy().decide("please calculate 3 * 4", [])
    assert decision.tool_calls[0].name == "calculator"


def test_heuristic_defaults_to_knowledge_search() -> None:
    decision = HeuristicPolicy().decide("explain langgraph", [])
    assert decision.tool_calls[0].name == "knowledge_search"
