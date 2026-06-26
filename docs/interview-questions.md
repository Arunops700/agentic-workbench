# Agent Interview Questions This Project Answers

Agent questions are the hardest section of 2026 interviews. These are grounded in this codebase.

---

### Q. What is ReAct and what problem does it solve over chain-of-thought?
ReAct interleaves **reasoning and acting**: the model thinks, calls a tool, observes the real result,
and repeats. Plain chain-of-thought reasons in one shot and will confidently *hallucinate* facts it
should have looked up. ReAct grounds each step in a real observation, so it can correct course. See
`react.py` for the loop.

### Q. Walk me through the anatomy of a production agent system.
Orchestrator (the loop/graph), a tool interface layer, memory, a policy/guardrails layer, and
observability. The LLM is maybe 20% of it — here it's deliberately behind the `Policy` seam; the
other 80% is the loop, the `ToolRegistry`, the checkpointer, and the step budget.

### Q. What are the main agent failure modes and how do you guard against them?
- **Runaway loops** → a **step budget** (`max_steps`) that hard-stops the loop. The #1 guard.
- **Tool misuse / bad arguments** → schema-described tools; the registry returns errors as *results*
  so the model can recover instead of crashing.
- **Unsafe tool execution** → never `eval()` model output; the calculator parses an AST.
- **Cost blowups** → step budget + (in production) a token/task budget and tracing.

### Q. Why LangGraph instead of a plain `while` loop?
The plain loop is fine until you need explicit persisted state, durable checkpoints, conditional
routing, pause/resume (human-in-the-loop), and observability. LangGraph models the agent as a state
machine that gives you those. This repo ships both so you can see the mapping (`react.py` ↔ `graph.py`).

### Q. How does the agent have memory?
A LangGraph **checkpointer** persists graph state per `thread_id`. A resumed run on the same thread
restores prior history — cross-turn memory — which is also the hook for human-in-the-loop interrupts.

### Q. How does tool calling actually work end to end?
You describe tools (name, description, JSON-schema params). The model emits a structured **tool-use**
request; **your code executes it** and returns a **tool-result**; the model continues. The model never
runs anything itself. `AnthropicPolicy` implements exactly this against Claude's API; `ToolRegistry`
executes and the loop feeds results back.

### Q. What is MCP and why does it matter?
The **Model Context Protocol** is an open standard for how AI apps expose tools, resources, and
prompts to models. A **server** publishes capabilities; any MCP-compatible **client/host** (Claude
Desktop, an IDE, our `mcp_client.py`) consumes them — no per-host glue. It decouples tool providers
from tool consumers, which is why it's becoming the interoperability layer for agents.

### Q. When should you NOT build an agent?
When the task is fully specifiable as a workflow (deterministic steps), when latency/cost can't
justify multi-step LLM calls, or when errors aren't recoverable. Reach for a single call or a
code-orchestrated workflow first; use an agent only when the trajectory genuinely can't be
pre-specified.

### Q. How would you add human-in-the-loop?
LangGraph supports interrupting before a node. You'd pause before an `act` step that calls a
destructive tool, surface the proposed call for approval, and resume from the checkpoint with the
decision — which is exactly why the agent already persists state per thread.
