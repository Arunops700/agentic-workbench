"""Command-line interface."""

from __future__ import annotations

from typing import Annotated

import typer

from agentic_workbench.config import load_settings
from agentic_workbench.factory import build_agent, build_policy
from agentic_workbench.react import run_react
from agentic_workbench.tools import default_registry

app = typer.Typer(help="A ReAct agent (LangGraph or from-scratch) with tools and an MCP server.")


@app.command()
def tools() -> None:
    """List the available tools."""
    for tool in default_registry()._tools.values():  # noqa: SLF001 - simple introspection
        typer.echo(f"{tool.name:18s} {tool.description}")


@app.command()
def run(
    task: Annotated[str, typer.Argument(help="The task for the agent.")],
    engine: Annotated[str, typer.Option(help="graph | react")] = "graph",
    thread: Annotated[str, typer.Option(help="Thread id (graph memory).")] = "default",
) -> None:
    """Run the agent on a task and print the answer and the steps it took."""
    settings = load_settings()
    if engine == "react":
        registry = default_registry()
        result = run_react(
            task=task,
            registry=registry,
            policy=build_policy(settings, registry),
            max_steps=settings.max_steps,
        )
    else:
        result = build_agent(settings).run(task, thread_id=thread)

    typer.echo(result.answer)
    if result.transcript:
        typer.echo("\nSteps:", err=True)
        for i, step in enumerate(result.transcript, start=1):
            for call, res in zip(step.tool_calls, step.results, strict=True):
                typer.echo(f"  {i}. {call.name}({call.args}) -> {res.output}", err=True)


@app.command(name="mcp-serve")
def mcp_serve() -> None:
    """Run the MCP server (stdio) exposing the agent's tools."""
    from agentic_workbench.mcp_server import main

    main()


@app.command(name="mcp-demo")
def mcp_demo() -> None:
    """Run a client that connects to the MCP server, lists tools, and calls one."""
    from agentic_workbench.mcp_client import main

    main()


if __name__ == "__main__":
    app()
