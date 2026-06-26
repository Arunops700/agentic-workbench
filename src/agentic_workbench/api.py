"""FastAPI service exposing the agent over HTTP."""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel, Field

from agentic_workbench.config import Settings, load_settings
from agentic_workbench.factory import build_agent
from agentic_workbench.tools import default_registry

app = FastAPI(title="agentic-workbench", version="0.1.0")


@lru_cache
def _settings() -> Settings:
    return load_settings()


class RunRequest(BaseModel):
    task: str = Field(min_length=1)
    thread_id: str = Field(default="default")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/tools")
def list_tools() -> list[dict[str, str]]:
    return [
        {"name": t.name, "description": t.description}
        for t in default_registry()._tools.values()  # noqa: SLF001
    ]


@app.post("/run")
def run(request: RunRequest) -> dict:
    result = build_agent(_settings()).run(request.task, thread_id=request.thread_id)
    return result.model_dump()
