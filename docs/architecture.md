# Architecture & Design Decisions

Why the agent is shaped the way it is. Read alongside the source.

## The agent loop (two implementations, one idea)

ReAct = **Reason + Act**: the policy proposes a tool call (or an answer), the loop runs it, feeds the
observation back, and repeats until the policy finishes.

- `react.py` — the loop with no framework, so the mechanics are visible. Read this first.
- `graph.py` — the same idea as a **LangGraph** state machine: a `reason` node and an `act` node,
  wired by a **conditional edge** (`reason → act` while there's work, `reason → END` when done).

```
reason ──tool calls──▶ act ──results──▶ reason ──finish──▶ END
```

## Key decisions

### 1. The `Policy` is a swappable seam — and it's what makes agents testable
Agents are notoriously hard to test because they call an LLM. We isolate the LLM behind a `Policy`
protocol (`decide(task, history) -> Decision`). Then:
- `ScriptedPolicy` returns a fixed sequence → **deterministic agents in tests**, no network, no spend.
- `HeuristicPolicy` → an offline rules-based brain so demos run with zero setup.
- `AnthropicPolicy` → real Claude tool-use.

The loop, memory, and guardrails are identical across all three, so they're tested once with the
scripted policy and trusted everywhere. **This testability is the central design point.**

### 2. One `ToolRegistry` feeds both the agent and the MCP server
A `Tool` couples a name + description + JSON schema (what the model sees) to a Python callable. The
registry **executes safely** — exceptions become `is_error` results the model can recover from, never
crashes the loop. The *same* registry is rendered into Anthropic tool schemas for the agent and
registered on the MCP server, so tools have a single source of truth.

### 3. From-scratch ReAct *and* LangGraph — on purpose
The hand-rolled loop teaches the mechanics and is trivial to reason about. LangGraph adds what you
need once an agent is real: explicit persisted state, a checkpointer, conditional routing, and
pause/resume (human-in-the-loop). Shipping both shows the concept and the production tool.

### 4. Memory via a checkpointer
LangGraph's `MemorySaver` persists state per `thread_id`. A `fresh=True` run seeds state; a
`fresh=False` run on the same thread omits history so the checkpointer restores it — genuine
cross-turn memory, and the foundation for human-in-the-loop interrupts.

### 5. Safety is not optional
Two guards interviewers look for:
- **Step budget** — the loop (and the graph's `reason` node) refuse to exceed `max_steps`. Without
  this, a confused agent loops forever and burns money. It's the single most important agent guard.
- **No `eval()` on model output** — the calculator parses an AST and evaluates only arithmetic. Tool
  inputs are untrusted model output; treat them that way.

### 6. MCP as a first-class surface
The Model Context Protocol standardizes how hosts consume tools. Publishing the registry on a
`FastMCP` server means Claude Desktop, an IDE, or our own client can use these tools with zero glue
per consumer. `mcp_client.py` demonstrates the round-trip (spawn server → initialize → list → call).

## Anatomy of a production agent (what this maps to)
Orchestrator (the loop/graph) · tool interface (registry) · memory (checkpointer) · policy/guardrails
(step budget, safe tools) · observability (transcript; tracing comes in Milestone 4). The LLM is one
part — deliberately behind the `Policy` seam.

## Trade-offs left open
- Multi-agent (supervisor + workers) — a LangGraph subgraph.
- Human-in-the-loop approval via `interrupt` before destructive tools.
- Tracing + prompt-injection guardrails — Milestone 4.
- `knowledge_search` calling the real deployed RAG service (Milestone 2) instead of a built-in KB.
