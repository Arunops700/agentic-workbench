"""Composition root: build a policy and agent from settings.

Falls back to the offline heuristic policy when no key is configured, so the agent runs end to end
with zero setup.
"""

from __future__ import annotations

from agentic_workbench.config import Settings
from agentic_workbench.graph import LangGraphAgent
from agentic_workbench.policy import AnthropicPolicy, HeuristicPolicy, Policy
from agentic_workbench.tools import ToolRegistry, default_registry


def build_policy(settings: Settings, registry: ToolRegistry) -> Policy:
    if settings.policy == "anthropic" and settings.anthropic_api_key:
        return AnthropicPolicy(
            registry=registry,
            model=settings.anthropic_model,
            max_tokens=settings.max_tokens,
            api_key=settings.anthropic_api_key,
        )
    return HeuristicPolicy()


def build_agent(settings: Settings) -> LangGraphAgent:
    registry = default_registry()
    policy = build_policy(settings, registry)
    return LangGraphAgent(registry=registry, policy=policy, max_steps=settings.max_steps)
