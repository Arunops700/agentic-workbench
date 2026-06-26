"""Domain exceptions."""

from __future__ import annotations


class AgentError(Exception):
    """Base class for agent failures."""


class ToolError(AgentError):
    """A tool failed or was called incorrectly. Returned to the model, not raised to the user."""


class PolicyError(AgentError):
    """The decision policy (LLM) failed."""


class MaxStepsError(AgentError):
    """The agent hit its step budget without finishing — the guard against runaway loops."""
