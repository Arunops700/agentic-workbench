"""The decision policy — the agent's "brain".

A `Policy` looks at the task and what's happened so far and decides the next step: finish, or call
tools. Separating it from the loop means the *same* loop runs deterministically in tests (with a
`ScriptedPolicy`), offline for demos (`HeuristicPolicy`), or for real (`AnthropicPolicy`, which uses
Claude's tool-use). This is the seam that makes agents testable without spending tokens.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Protocol, cast

from agentic_workbench.errors import PolicyError
from agentic_workbench.models import Decision, Step, ToolCall

if TYPE_CHECKING:
    from agentic_workbench.tools import ToolRegistry


class Policy(Protocol):
    def decide(self, task: str, history: list[Step]) -> Decision: ...


class ScriptedPolicy:
    """Returns a pre-programmed sequence of decisions — deterministic agents for tests."""

    def __init__(self, decisions: list[Decision]) -> None:
        self._decisions = decisions

    def decide(self, task: str, history: list[Step]) -> Decision:
        idx = len(history)
        if idx < len(self._decisions):
            return self._decisions[idx]
        # Ran off the end of the script — finish to avoid an infinite loop.
        return Decision(finish="(scripted policy exhausted)")


class HeuristicPolicy:
    """A rules-based offline policy: one tool call, then summarize. No API key required."""

    def decide(self, task: str, history: list[Step]) -> Decision:
        if history:  # we already ran a tool; answer from its result
            last = history[-1].results[-1].output if history[-1].results else ""
            return Decision(finish=f"Result: {last}")

        cid = "call-0"
        if re.search(r"\d\s*[-+*/]\s*\d|\bcalcul", task.lower()):
            expr = task
            match = re.search(r"[-+*/()\d.\s]+", task)
            if match:
                expr = match.group().strip()
            return Decision(
                tool_calls=[ToolCall(id=cid, name="calculator", args={"expression": expr})]
            )
        if "word" in task.lower() and "count" in task.lower():
            return Decision(tool_calls=[ToolCall(id=cid, name="word_count", args={"text": task})])
        return Decision(
            tool_calls=[ToolCall(id=cid, name="knowledge_search", args={"query": task})]
        )


_SYSTEM = (
    "You are a capable assistant with tools. Think step by step and call a tool when it helps. "
    "When you have enough information, answer the user directly and concisely."
)


class AnthropicPolicy:
    """Real tool-use decisions via Claude; reconstructs the message history each call (stateless)."""  # noqa: E501

    def __init__(self, *, registry: ToolRegistry, model: str, max_tokens: int, api_key: str | None):
        self._registry = registry
        self._model = model
        self._max_tokens = max_tokens
        self._api_key = api_key

    def _messages(self, task: str, history: list[Step]) -> list[dict]:
        messages: list[dict] = [{"role": "user", "content": task}]
        for step in history:
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": c.id, "name": c.name, "input": c.args}
                        for c in step.tool_calls
                    ],
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": r.id,
                            "content": r.output,
                            "is_error": r.is_error,
                        }
                        for r in step.results
                    ],
                }
            )
        return messages

    def decide(self, task: str, history: list[Step]) -> Decision:
        import anthropic

        try:
            response = anthropic.Anthropic(api_key=self._api_key).messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=_SYSTEM,
                tools=cast(Any, self._registry.to_anthropic_tools()),
                messages=cast(Any, self._messages(task, history)),
            )
        except anthropic.APIError as exc:
            raise PolicyError(f"Anthropic request failed: {exc}") from exc

        tool_calls = [
            ToolCall(id=b.id, name=b.name, args=dict(b.input))
            for b in response.content
            if b.type == "tool_use"
        ]
        if tool_calls:
            return Decision(tool_calls=tool_calls)
        text = "".join(b.text for b in response.content if b.type == "text")
        return Decision(finish=text or "(no answer)")
