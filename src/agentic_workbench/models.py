"""Core agent types.

The agent loop speaks in three nouns: a `ToolCall` (the model wants to run a tool), a `ToolResult`
(what running it produced), and a `Decision` (the policy's choice this step: finish, or call
tools). Keeping these explicit is what makes both the ReAct loop and the LangGraph version readable.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    id: str = Field(description="Unique id linking a call to its result.")
    name: str = Field(description="Tool name.")
    args: dict = Field(default_factory=dict, description="Tool arguments.")


class ToolResult(BaseModel):
    id: str = Field(description="Matches the originating ToolCall id.")
    name: str
    output: str
    is_error: bool = False


class Step(BaseModel):
    """One reasoning step: the tools the policy called and what they returned."""

    tool_calls: list[ToolCall] = Field(default_factory=list)
    results: list[ToolResult] = Field(default_factory=list)


class Decision(BaseModel):
    """A policy's output for one step.

    Exactly one of these is meaningful: if `finish` is set, the agent answers and stops; otherwise
    it executes `tool_calls` and loops. This explicit shape is what the conditional edge in the
    LangGraph routes on.
    """

    finish: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)

    @property
    def is_final(self) -> bool:
        return self.finish is not None


class AgentResult(BaseModel):
    """The outcome of a full agent run."""

    task: str
    answer: str
    steps: int
    transcript: list[Step] = Field(default_factory=list)
