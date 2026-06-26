"""HTTP surface, using the offline heuristic policy (no keys)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from agentic_workbench import api

client = TestClient(api.app)


def test_health() -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_list_tools() -> None:
    names = {t["name"] for t in client.get("/tools").json()}
    assert {"calculator", "knowledge_search", "word_count"} <= names


def test_run_returns_answer() -> None:
    resp = client.post("/run", json={"task": "what is mcp?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"]
    assert "transcript" in body


def test_run_rejects_empty_task() -> None:
    assert client.post("/run", json={"task": ""}).status_code == 422
