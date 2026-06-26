"""The from-scratch ReAct loop: tool execution, finishing, and the step-budget guard."""

from __future__ import annotations

import pytest

from agentic_workbench.errors import MaxStepsError
from agentic_workbench.models import Decision, Step, ToolCall
from agentic_workbench.policy import ScriptedPolicy
from agentic_workbench.react import run_react
from agentic_workbench.tools import default_registry
from tests.conftest import calc, finish


def test_runs_tool_then_finishes() -> None:
    policy = ScriptedPolicy([calc("2+2"), finish("The answer is 4.")])
    result = run_react(task="add 2 and 2", registry=default_registry(), policy=policy)

    assert result.answer == "The answer is 4."
    assert result.steps == 2
    assert len(result.transcript) == 1
    assert result.transcript[0].results[0].output == "4.0"


def test_step_budget_prevents_runaway() -> None:
    """A policy that never finishes must be stopped by the step budget, not loop forever."""

    class AlwaysTool:
        def decide(self, task: str, history: list[Step]) -> Decision:
            cid = f"c{len(history)}"
            return Decision(tool_calls=[ToolCall(id=cid, name="word_count", args={"text": "x"})])

    with pytest.raises(MaxStepsError):
        run_react(task="loop", registry=default_registry(), policy=AlwaysTool(), max_steps=3)
