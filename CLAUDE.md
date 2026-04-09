# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the App

```bash
# Primary interface — Streamlit dashboard (multi-user, live monitoring)
streamlit run dashboard.py

# CLI runner — headless HITL loop in the terminal
python main.py "Your research topic here"

# REST API — FastAPI server for programmatic HITL
uvicorn api:app --reload

# LangGraph Studio — visual graph debugger
langgraph dev
```

Required `.env` file in this directory:
```
GOOGLE_AISTUDIO_API_KEY=...   # for Gemini provider
TAVILY_API_KEY=...             # for web search
OPENROUTER_API_KEY=...         # optional — for OpenRouter provider
```

## Architecture

The system is a 6-node LangGraph pipeline that researches a topic and writes a blog post, with 3 human-in-the-loop (HITL) checkpoints where execution pauses for human review.

### Data flow

```
[research_agent] → [review_queries] ⏸ → [web_search_node] → [writer_agent] → [review_draft] ⏸ → [publisher] ⏸
                         ↓ reject                                    ↑ revise ↙
                       __end__                                   (revision loop)
```

HITL checkpoints use LangGraph's `interrupt()` / `Command(resume=value)` pattern. The graph state is checkpointed by `SqliteSaver` (`checkpoints.db`) after each node — resuming is done by passing `Command(resume=human_response)` back to `graph.invoke()` with the same `thread_id`.

Dynamic routing happens inside nodes via `Command(goto=...)` — `review_queries` can go to `__end__` on reject; `review_draft` can go to `publisher` (approve/edit), loop back to `writer_agent` (revise), or `__end__` (reject). No conditional edges are needed in `graph.py`.

### File responsibilities

| File | Role |
|------|------|
| `state.py` | `PipelineState` TypedDict — single source of truth for all graph state fields |
| `graph.py` | Builds and compiles the `StateGraph`; imports all nodes from `agents.py` |
| `agents.py` | The 6 node functions; calls `get_llm()` / `get_llm_json()` from `llm_config` |
| `llm_config.py` | Switchable LLM provider (Gemini API or Ollama); module-level `_config` dict persists across Streamlit reruns via `sys.modules` cache |
| `tools.py` | `web_search` (Tavily), `publish_to_platform` (mock), `parse_json_list` |
| `dashboard.py` | Streamlit multi-user dashboard; runs graph in background threads; persists sessions to SQLite |
| `main.py` | Terminal HITL runner — simplest way to exercise the pipeline end-to-end |
| `api.py` | FastAPI REST interface for the same pipeline |

### LLM provider switching

`llm_config.py` is the only place LLM instances are created. `agents.py` calls `get_llm()` / `get_llm_json()` on each invocation (not at import time), so switching providers in the dashboard takes effect immediately for the next pipeline run. LLM instances are cached by `(provider, model, json_mode)` key in `_cache`.

- **Gemini** (default): uses `GOOGLE_AISTUDIO_API_KEY`; model `gemini-2.5-flash`; JSON mode via `response_mime_type="application/json"`
- **Ollama**: requires `ollama serve` running locally on port 11434; model selectable in UI
- **OpenRouter**: uses `OPENROUTER_API_KEY`; proxies to any model (GPT-4o, Claude, Llama, etc.) via OpenAI-compatible API at `https://openrouter.ai/api/v1`

### Dashboard internals

`dashboard.py` uses `@st.cache_resource` to hold a singleton registry dict (`_SESSIONS`, `_QUEUES`, `_LOCK`) that survives Streamlit reruns. Each pipeline run executes in a background `threading.Thread`; the thread communicates with the UI thread via a `queue.Queue` (one per session). HITL responses are sent from the UI into the queue, which the background thread reads to call `graph.invoke(Command(resume=...))`.

Two SQLite databases are used:
- `checkpoints.db` — LangGraph's `SqliteSaver` checkpointer; stores full graph state after every node so pipelines survive restarts
- `sessions.db` — dashboard's own session registry; stores metadata, status, `hitl_events`, and timing for the UI

User preferences (name, LLM provider, Ollama model) are saved to `.dashboard_prefs.json`.

`.streamlit/config.toml` sets `fileWatcherType = "poll"`, blacklists the `torch` folder from file-watching (prevents noisy tracebacks on reload), and sets `logger.level = "error"` to suppress Streamlit info/warning log spam.

### TAPO pattern

Each agent node follows **Thought → Action → Pause → Observation**:
- *Thought*: embedded in the system prompt — tells the LLM how to reason
- *Action*: the node function body that produces a state update
- *Pause*: `interrupt(payload)` — halts the graph and checkpoints state
- *Observation*: `Command(resume=value)` — human's response fed back in on resume
