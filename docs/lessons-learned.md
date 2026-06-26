# Lessons Learned

Notes to my future self from building this (Milestone 3).

## Technical
- **The `Policy` seam is everything.** Putting the LLM behind a `decide(task, history)` protocol made
  the agent deterministic and free to test. A `ScriptedPolicy` turns "test an agent" from a hard,
  flaky, costly problem into ordinary unit testing. If I take one idea forward, it's this.
- **Build ReAct from scratch before reaching for a framework.** Writing the loop by hand made
  LangGraph's nodes/edges/checkpointer obvious instead of magic — and made it clear which problems
  the framework actually solves (state, checkpoints, routing, pause/resume).
- **One tool registry, many consumers.** Feeding the same `ToolRegistry` to the agent *and* the MCP
  server meant tools have a single source of truth. Adding a tool lights it up everywhere.
- **Verify framework APIs against the installed version.** I probed LangGraph 1.2 and MCP 1.28
  (`StateGraph`, `MemorySaver`, `FastMCP.list_tools`) before writing the modules — agent frameworks
  move fast, and coding from memory would have shipped bugs.
- **Safety is a feature, not a footnote.** The step budget and the AST-based calculator (never
  `eval()`) are small but exactly what interviewers probe — and what stops real incidents.

## Process
- **Make it runnable offline first.** Heuristic + scripted policies meant the CLI, API, and tests all
  work with zero keys — fast iteration and a demo anyone can run.
- **Library first, surfaces thin.** CLI and API are shells over `build_agent(...)`; all logic and
  tests live in the library.

## If I did it again
- Add human-in-the-loop (`interrupt`) and a multi-agent supervisor subgraph from the start.
- Wire `knowledge_search` to the deployed RAG service (Milestone 2) for a real cross-project tool.
- Add tracing now (Milestone 4 will retrofit it) so agent runs are observable, not just printable.
