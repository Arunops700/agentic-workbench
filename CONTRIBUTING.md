# Contributing

## Setup
```bash
uv sync --extra dev
uv run pre-commit install
```

## Checks (CI enforces these)
```bash
uv run ruff check .
uv run ruff format .
uv run mypy .
uv run pytest
```

## Conventions
- Type hints everywhere; mypy clean. Tests for new logic, driven by `ScriptedPolicy` (no network).
- New tool: add it to `tools.py` and register it in `default_registry()` — the agent *and* the MCP
  server pick it up automatically.
- New decision strategy: implement the `Policy` protocol and wire it in `factory.py`.
- Secrets via `.env` (never committed); update `.env.example` when adding a variable.
- Conventional-commit messages (`feat:`, `fix:`, `docs:`, `test:`, `chore:`).
