"""A from-scratch ReAct loop — no framework, so the mechanics are visible.

ReAct = Reason + Act: the policy proposes a tool call (or an answer), the loop executes it, feeds
the observation back, and repeats. The single most important production guard is the **step
budget**: without it, a confused agent loops forever and burns money. The LangGraph version
(`graph.py`) implements the same idea as a state machine; read this first to see what it does.
"""

from __future__ import annotations

from agentic_workbench.errors import MaxStepsError
from agentic_workbench.models import AgentResult, Step
from agentic_workbench.policy import Policy
from agentic_workbench.tools import ToolRegistry


def run_react(
    *, task: str, registry: ToolRegistry, policy: Policy, max_steps: int = 8
) -> AgentResult:
    """Run the agent until the policy finishes or the step budget is exhausted."""
    history: list[Step] = []
    for step_no in range(1, max_steps + 1):
        decision = policy.decide(task, history)
        if decision.is_final:
            return AgentResult(
                task=task, answer=decision.finish or "", steps=step_no, transcript=history
            )
        # Execute every requested tool (the model may call several in one step), collect results.
        results = [registry.execute(call) for call in decision.tool_calls]
        history.append(Step(tool_calls=decision.tool_calls, results=results))

    raise MaxStepsError(f"Agent did not finish within {max_steps} steps.")
