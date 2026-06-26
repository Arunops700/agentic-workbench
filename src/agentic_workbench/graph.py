"""The LangGraph agent — the same ReAct idea as a state machine, with memory.

LangGraph models the agent as a graph: a **reason** node (the policy decides) and an **act** node
(tools run), wired with a **conditional edge** that routes reason → act while there's work, or
reason → END when the policy finishes. A **checkpointer** persists state per `thread_id`, giving
the agent memory across turns and forming the foundation for human-in-the-loop.

Why a graph instead of the plain loop in `react.py`? Explicit state, durable checkpoints,
conditional routing, and pause/resume — the things you need once an agent is more than a demo.
"""

from __future__ import annotations

from operator import add
from typing import Annotated, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from agentic_workbench.models import AgentResult, Step, ToolCall
from agentic_workbench.policy import Policy
from agentic_workbench.tools import ToolRegistry


class AgentState(TypedDict):
    """Graph state. `history` and `steps` accumulate (reducers); the rest are last-write."""

    task: str
    history: Annotated[list[Step], add]
    pending: list[ToolCall]
    answer: str | None
    steps: Annotated[int, add]


class LangGraphAgent:
    """A compiled ReAct graph with a checkpointer for per-thread memory."""

    def __init__(self, *, registry: ToolRegistry, policy: Policy, max_steps: int = 8) -> None:
        self._registry = registry
        self._policy = policy
        self._max_steps = max_steps
        self._app = self._build().compile(checkpointer=MemorySaver())

    def _reason(self, state: AgentState) -> dict:
        # Safety guard: never exceed the step budget, even if the policy keeps asking for tools.
        if len(state["history"]) >= self._max_steps:
            return {"answer": "(stopped: reached max steps)", "pending": [], "steps": 1}
        decision = self._policy.decide(state["task"], state["history"])
        if decision.is_final:
            return {"answer": decision.finish, "pending": [], "steps": 1}
        return {"pending": decision.tool_calls, "steps": 1}

    def _act(self, state: AgentState) -> dict:
        pending = state["pending"]
        results = [self._registry.execute(call) for call in pending]
        return {"history": [Step(tool_calls=pending, results=results)], "pending": []}

    @staticmethod
    def _route(state: AgentState) -> str:
        return "act" if state.get("pending") else "end"

    def _build(self) -> StateGraph:
        graph = StateGraph(AgentState)
        graph.add_node("reason", self._reason)
        graph.add_node("act", self._act)
        graph.set_entry_point("reason")
        graph.add_conditional_edges("reason", self._route, {"act": "act", "end": END})
        graph.add_edge("act", "reason")
        return graph

    def run(self, task: str, *, thread_id: str = "default", fresh: bool = True) -> AgentResult:
        """Run the agent. With the same `thread_id` and `fresh=False`, prior history persists."""
        config: RunnableConfig = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 2 * self._max_steps + 5,
        }
        # Fresh runs seed full state; resumed runs omit history/steps so the checkpointer restores
        # them (cross-turn memory). `input` is typed Any by LangGraph, so a plain dict is fine.
        init: dict
        if fresh:
            init = {"task": task, "history": [], "pending": [], "answer": None, "steps": 0}
        else:
            init = {"task": task, "pending": [], "answer": None}
        final = self._app.invoke(init, config)
        return AgentResult(
            task=task,
            answer=final.get("answer") or "",
            steps=final.get("steps", 0),
            transcript=final.get("history", []),
        )
