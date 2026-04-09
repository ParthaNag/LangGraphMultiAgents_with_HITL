# Building Multi-Agent HITL Systems with LangGraph

**LangGraph's `interrupt()` function, introduced in December 2024, is the cleanest way to pause an AI agent workflow, collect human feedback, and resume execution exactly where it stopped.** Combined with `Command(resume=value)` and a checkpointer for state persistence, it enables multi-agent systems where humans gate critical decisions without breaking the flow. This guide walks through the architecture, mechanisms, TAPO prompting strategy, and a complete project build using LangGraph's latest patterns.

The project we'll build — a **Content Research & Publishing Pipeline** — chains a research agent, a writing agent, and a publishing agent together with human review checkpoints between each stage. It demonstrates three distinct HITL patterns (query approval, content editing, publish confirmation) while staying simple enough to understand and extend.

---

## Quick Start

**Install dependencies:**

```bash
pip install -r requirements.txt
```

**Create a `.env` file in the project root:**

```
GOOGLE_AISTUDIO_API_KEY=...   # for Gemini (default provider)
TAVILY_API_KEY=...             # for web search
OPENROUTER_API_KEY=...         # optional — for OpenRouter provider
```

**Run the Streamlit dashboard** (recommended — multi-user, live monitoring, HITL UI):

```bash
streamlit run dashboard.py
```

**Or run headless from the terminal:**

```bash
python main.py "Your research topic here"
```

**Or start the FastAPI server for programmatic HITL:**

```bash
uvicorn api:app --reload
```

**LLM providers** — switch in the dashboard sidebar or via `llm_config.set_provider()`:

| Provider | Key Required | Notes |
|---|---|---|
| `gemini` (default) | `GOOGLE_AISTUDIO_API_KEY` | `gemini-2.5-flash`; JSON mode via `response_mime_type` |
| `ollama` | none | Requires `ollama serve` on port 11434 |
| `openrouter` | `OPENROUTER_API_KEY` | Access to GPT-4o, Claude, Llama, Mistral, etc. |

State is persisted across restarts via two SQLite databases: `checkpoints.db` (LangGraph graph state) and `sessions.db` (dashboard session registry).

---

## How `interrupt()` Pauses and Resumes a Graph

LangGraph's HITL system rests on three primitives that work together: the `interrupt()` function pauses execution, the checkpointer saves state, and `Command(resume=value)` restarts from exactly where the graph stopped.

**The `interrupt()` function** works by raising a special exception inside a node. The LangGraph runtime catches it, serializes the current graph state via the checkpointer, and surfaces the interrupt payload to the caller. The payload — any JSON-serializable value — acts as the "question" presented to the human. Unlike older static breakpoints (`interrupt_before`/`interrupt_after`), `interrupt()` can be placed **anywhere inside node code**, including inside conditionals, loops, and even tool functions.

```python
from langgraph.types import interrupt, Command

def human_review_node(state: State):
    # This pauses the entire graph and surfaces the draft to the human
    decision = interrupt({
        "question": "Approve this draft?",
        "content": state["draft_content"]
    })
    # When resumed, `decision` holds the value from Command(resume=...)
    if decision["action"] == "approve":
        return Command(goto="publish", update={"status": "approved"})
    elif decision["action"] == "edit":
        return Command(goto="rewrite", update={"draft_content": decision["edited_text"]})
    return Command(goto="end", update={"status": "rejected"})
```

**Resuming with `Command(resume=value)`** is straightforward. You invoke the graph again with the same `thread_id` and pass the human's response as the resume value. That value becomes the return value of the `interrupt()` call inside the paused node:

```python
from langgraph.types import Command

config = {"configurable": {"thread_id": "thread-1"}}

# Initial run — hits interrupt, pauses, returns interrupt info
result = graph.invoke({"topic": "AI in healthcare"}, config=config)
print(result["__interrupt__"])  # Shows the interrupt payload

# Human decides to approve → resume
graph.invoke(Command(resume={"action": "approve"}), config=config)
```

**The checkpointer** is what makes this possible. Without it, the graph has no memory of where it stopped. `MemorySaver` (an alias for `InMemorySaver`) works for development; **PostgresSaver** or **RedisSaver** should be used in production. The checkpointer stores the full graph state at every super-step boundary, keyed by `thread_id`.

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
```

| Checkpointer | Import | Best For |
|---|---|---|
| `MemorySaver` | `langgraph.checkpoint.memory` | Development/testing |
| `SqliteSaver` | `langgraph.checkpoint.sqlite` | Lightweight persistence |
| `PostgresSaver` | `langgraph.checkpoint.postgres` | Production workloads |
| `RedisSaver` | `langgraph.checkpoint.redis` | High-performance production |

---

## Designing Agent Prompts with the TAPO Cycle

The most effective prompting pattern for multi-agent HITL systems follows the **Thought → Action → Pause → Observation** loop. Each agent node in LangGraph should be prompted to reason through this cycle explicitly. This isn't just a theoretical framework — it maps directly onto LangGraph's execution model and produces dramatically better agent behavior.

### How TAPO Maps to LangGraph

Each step of the TAPO cycle has a concrete LangGraph implementation:

**Thought** is the agent's internal reasoning before it acts. In code, this is the chain-of-thought section of the system prompt passed to the LLM inside the node function. The agent should articulate *what* it plans to do, *why*, and *what risks or edge cases* to watch for. This step happens entirely inside the LLM call — no tools are invoked yet.

**Action** is the agent executing its plan — calling tools, generating content, transforming state. In LangGraph, this is the body of the node function that returns a state update dictionary. The key discipline here is that the Action should be *scoped* — do one coherent thing and do it well, rather than trying to accomplish the entire pipeline in a single node.

**Pause** is `interrupt()`. The graph halts, the checkpointer serializes the full state, and the interrupt payload (containing the agent's output) is surfaced to the human. The human can now review, edit, approve, or reject the agent's work. This is the moment where human judgment enters the loop.

**Observation** is `Command(resume=value)`. The human's feedback becomes the return value of the `interrupt()` call. The agent absorbs this feedback and either proceeds to the next phase or loops back for revision. The observation from one TAPO cycle becomes the context for the *next* cycle's Thought step.

#### Diagram: TAPO Cycle → LangGraph Mapping

```mermaid
graph LR
    subgraph TAPO["TAPO Prompting Cycle"]
        direction TB
        T["🧠 <b>THOUGHT</b><br/>Agent reasons about task<br/><i>'I need 3 queries on AI agents'</i>"]
        A["⚡ <b>ACTION</b><br/>Agent executes the plan<br/><i>Generates queries / writes draft</i>"]
        P["⏸️ <b>PAUSE</b><br/>Graph halts for human review<br/><i>State checkpointed to storage</i>"]
        O["👁️ <b>OBSERVATION</b><br/>Human feedback received<br/><i>Agent resumes with context</i>"]
        T --> A --> P --> O
        O -.->|"Next agent<br/>cycle"| T
    end

    subgraph LG["LangGraph Mechanism"]
        direction TB
        L1["System prompt with<br/>chain-of-thought instructions"]
        L2["Node function body<br/>returns state update dict"]
        L3["<code>interrupt(payload)</code><br/>Checkpointer saves full state"]
        L4["<code>Command(resume=value)</code><br/>Human value → interrupt return"]
        L1 ~~~ L2 ~~~ L3 ~~~ L4
    end

    T -.- L1
    A -.- L2
    P -.- L3
    O -.- L4

    style T fill:#EEEDFE,stroke:#534AB7,color:#3C3489
    style A fill:#E1F5EE,stroke:#0F6E56,color:#085041
    style P fill:#FAECE7,stroke:#993C1D,color:#712B13
    style O fill:#FAEEDA,stroke:#854F0B,color:#633806
    style L1 fill:#F1EFE8,stroke:#5F5E5A,color:#444441
    style L2 fill:#F1EFE8,stroke:#5F5E5A,color:#444441
    style L3 fill:#F1EFE8,stroke:#5F5E5A,color:#444441
    style L4 fill:#F1EFE8,stroke:#5F5E5A,color:#444441
```

### Structuring System Prompts for TAPO

The key to making TAPO work is embedding it into the system prompt so the LLM naturally follows the cycle. Here's the pattern for each agent in the pipeline:

**Research Agent system prompt:**
```
You are a research planning agent. Follow this reasoning process:

THOUGHT: Analyze the topic "{topic}". Consider:
- What are the key subtopics that need investigation?
- What would a domain expert search for?
- What gaps might exist in obvious search queries?

ACTION: Generate exactly 3 focused search queries that will produce
high-quality, complementary research results. Each query should
target a different angle of the topic.

Return your queries as a JSON list. Include a brief rationale for
each query choice.
```

**Writer Agent system prompt:**
```
You are a content writing agent. Follow this reasoning process:

THOUGHT: Review the research results provided. Consider:
- What is the central narrative these findings support?
- What claims are well-supported vs. speculative?
- What structure will best serve the reader?

ACTION: Write a complete blog post that synthesizes the research
into a coherent narrative. Use specific evidence from the research
to support each key point.

Format: Title, introduction, 3-4 body sections, conclusion.
Tone: Authoritative but accessible.
```

**Publisher Agent system prompt:**
```
You are a content publishing agent. Follow this reasoning process:

THOUGHT: Review the finalized content. Consider:
- Is the formatting correct for the target platform?
- Are all links, references, and attributions in place?
- Does the opening hook work for the intended audience?

ACTION: Prepare the content for publication. Add metadata (tags,
description, category) and format for the target platform.
Present a final summary for human confirmation.
```

### How the Human's Observation Feeds Back

The critical insight is that **Pause** and **Observation** aren't passive — the human's response during the Pause step shapes the next Thought. Here's how each HITL checkpoint uses the TAPO loop:

**HITL #1 (Query Review)** — The Research Agent completes Thought + Action by generating search queries. The system Pauses via `interrupt()`, presenting the queries to the human. The human's Observation ("approve as-is" / "edit to these new queries" / "reject") becomes the input for the Web Search node's execution. If the human edits the queries, the modified queries replace the original ones in state — the agent's Action gets refined by human judgment.

**HITL #2 (Draft Review)** — The Writer Agent completes Thought + Action by producing a draft. The system Pauses, presenting the full draft to the human. The human's Observation has three paths: approval (proceed to publishing), direct editing (human modifies the text and it moves forward), or revision request (the graph loops back to the Writer Agent, which now runs a *new* TAPO cycle with the human's feedback as additional context in its Thought step).

**HITL #3 (Publish Confirmation)** — The Publisher Agent completes Thought + Action by formatting content and metadata. The system Pauses for a final go/no-go. The human's Observation is binary — confirm publication or cancel. This is the lightest HITL checkpoint, but it's the most consequential because it's the last gate before an irreversible external action.

### The Revision Loop: TAPO Within TAPO

When the human requests a revision at HITL #2, the Writer Agent enters a *nested* TAPO cycle. Its new Thought step now includes both the original research *and* the human's feedback. The system prompt for the revision round should explicitly reference what changed:

```
THOUGHT: The previous draft was returned for revision.
Human feedback: "{feedback}"
Consider what specific changes are needed and why the previous
version fell short.

ACTION: Rewrite the draft addressing the feedback while preserving
the strengths of the original version.
```

This creates a tightening spiral where each TAPO cycle produces progressively better output. The human stays in the lead — they set the quality bar, the agent does the heavy lifting, and the `interrupt()` mechanism ensures nothing proceeds without approval.

### TAPO as a Debugging Tool

When an agent produces poor output, trace back through the TAPO cycle to find where it broke down:

- Was the **Thought** step insufficiently scoped? → Improve the system prompt reasoning instructions.
- Did the **Action** step miss something? → Add tool capabilities or better context.
- Was the **Pause** step at the wrong point? → Move the `interrupt()` call.
- Did the **Observation** not carry enough signal? → Enrich the interrupt payload so the human has better information to act on.

This structured diagnosis is one of the biggest practical benefits of designing around TAPO from the start.

---

## The Project: Content Research & Publishing Pipeline

The project chains three specialized agents — **Research**, **Writer**, and **Publisher** — into a sequential workflow with human review gates between each stage. A human approves search queries before research begins, reviews the draft before it's finalized, and confirms before the content is published. This mirrors real editorial workflows and demonstrates all major HITL patterns.

**The architecture:**

```
START → research_agent → [HITL #1: review queries] → web_search
      → writer_agent → [HITL #2: review draft] → publisher_agent
      → [HITL #3: confirm publish] → END
```

#### Diagram: Pipeline Architecture with HITL Checkpoints

```mermaid
graph TD
    START(["⬤ Start"]) --> RA

    subgraph PHASE1["Phase 1: Research"]
        RA["🧠 <b>Research Agent</b><br/><i>Generates search queries</i><br/>TAPO: Thought + Action"]
        RA --> H1
        H1{"🛑 <b>HITL #1</b><br/>Review Queries<br/><code>interrupt()</code><br/>TAPO: Pause"}
        H1 -->|"✅ Approve / ✏️ Edit<br/>TAPO: Observation"| WS
        WS["🔍 <b>Web Search</b><br/><i>Executes approved queries</i>"]
    end

    subgraph PHASE2["Phase 2: Writing"]
        WA["🧠 <b>Writer Agent</b><br/><i>Drafts content from research</i><br/>TAPO: Thought + Action"]
        WA --> H2
        H2{"🛑 <b>HITL #2</b><br/>Review Draft<br/><code>interrupt()</code><br/>TAPO: Pause"}
        H2 -->|"✅ Approve / ✏️ Edit<br/>TAPO: Observation"| PUB
        H2 -.->|"🔄 Revise<br/>Feedback → next Thought"| WA
    end

    subgraph PHASE3["Phase 3: Publishing"]
        PUB["🧠 <b>Publisher Agent</b><br/><i>Formats final content</i><br/>TAPO: Thought + Action"]
        PUB --> H3
        H3{"🛑 <b>HITL #3</b><br/>Confirm Publish<br/><code>interrupt()</code><br/>TAPO: Pause"}
    end

    WS --> WA
    H3 -->|"✅ Confirm<br/>TAPO: Observation"| DONE(["⬤ End"])
    H1 -->|"❌ Reject"| DONE
    H2 -->|"❌ Reject"| DONE
    H3 -->|"❌ Cancel"| DONE

    style RA fill:#EEEDFE,stroke:#534AB7,color:#3C3489
    style WA fill:#EEEDFE,stroke:#534AB7,color:#3C3489
    style PUB fill:#EEEDFE,stroke:#534AB7,color:#3C3489
    style WS fill:#E1F5EE,stroke:#0F6E56,color:#085041
    style H1 fill:#FAECE7,stroke:#993C1D,color:#712B13
    style H2 fill:#FAECE7,stroke:#993C1D,color:#712B13
    style H3 fill:#FAECE7,stroke:#993C1D,color:#712B13
    style START fill:#F1EFE8,stroke:#5F5E5A,color:#444441
    style DONE fill:#F1EFE8,stroke:#5F5E5A,color:#444441
    style PHASE1 fill:transparent,stroke:#B4B2A9,stroke-dasharray:5 5
    style PHASE2 fill:transparent,stroke:#B4B2A9,stroke-dasharray:5 5
    style PHASE3 fill:transparent,stroke:#B4B2A9,stroke-dasharray:5 5
```

**Legend:**
- 🟣 Purple nodes = Agent nodes (LLM-powered, TAPO Thought + Action)
- 🟠 Coral diamonds = Human review checkpoints (TAPO Pause + Observation)
- 🟢 Teal node = Tool execution node
- ⬜ Gray nodes = Start / End
- Dashed arrow = Revision feedback loop (Observation → next Thought)

Each HITL checkpoint uses `interrupt()` to pause, present output to the human, and branch based on their response. The state flows through a single `TypedDict` that all nodes read from and write to.

### Step 1: Define the Typed State

The state schema is the backbone. Every node reads from it and returns partial updates. Use `Annotated` fields with reducers when multiple nodes append to the same field.

```python
from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages
import operator

class PipelineState(TypedDict):
    topic: str
    search_queries: list[str]
    research_results: Annotated[list[str], operator.add]  # appends across nodes
    draft_content: str
    final_content: str
    status: str
    human_feedback: str  # carries TAPO Observation into next cycle
    messages: Annotated[list, add_messages]  # smart message merging
```

The `Annotated[list[str], operator.add]` reducer means when multiple nodes return `research_results`, the lists get concatenated rather than overwritten. The `add_messages` reducer intelligently merges message lists, handling deduplication by ID. The `human_feedback` field explicitly carries the Observation from one TAPO cycle into the next Thought step.

### Step 2: Build the Agent Nodes (with TAPO Prompts)

Each node is a plain Python function that receives the full state and returns only the fields it modifies. Notice how the system prompts embed the TAPO Thought/Action structure:

```python
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt, Command
from typing import Literal

llm = ChatOpenAI(model="gpt-4o")

# --- Node 1: Research Agent (TAPO: Thought + Action) ---
def research_agent(state: PipelineState):
    # TAPO Thought + Action embedded in the system prompt
    feedback_context = ""
    if state.get("human_feedback"):
        feedback_context = f"\nPrevious feedback: {state['human_feedback']}"

    response = llm.invoke(
        f"""You are a research planning agent.

THOUGHT: Analyze the topic "{state['topic']}".{feedback_context}
Consider what a domain expert would search for and what gaps
might exist in obvious queries.

ACTION: Generate exactly 3 focused, complementary search queries.
Return as a JSON list of strings with a brief rationale for each."""
    )
    queries = parse_json_list(response.content)
    return {"search_queries": queries}

# --- Node 2: Human Reviews Queries (TAPO: Pause + Observation) ---
def review_queries(state: PipelineState):
    # TAPO Pause: interrupt surfaces queries for human review
    decision = interrupt({
        "step": "review_queries",
        "tapo_phase": "PAUSE — awaiting human observation",
        "suggested_queries": state["search_queries"],
        "instruction": "Approve, edit, or reject these search queries."
    })
    # TAPO Observation: human's response feeds back into state
    if decision["action"] == "approve":
        return {"search_queries": state["search_queries"],
                "status": "queries_approved",
                "human_feedback": ""}
    elif decision["action"] == "edit":
        return {"search_queries": decision["queries"],
                "status": "queries_edited",
                "human_feedback": decision.get("reason", "")}
    return Command(goto="__end__", update={"status": "cancelled"})

# --- Node 3: Execute Web Search ---
def web_search(state: PipelineState):
    results = []
    for query in state["search_queries"]:
        result = tavily_client.search(query=query, max_results=3)
        results.extend([r["content"] for r in result["results"]])
    return {"research_results": results}

# --- Node 4: Writer Agent (TAPO: Thought + Action) ---
def writer_agent(state: PipelineState):
    context = "\n\n".join(state["research_results"])
    feedback_context = ""
    if state.get("human_feedback"):
        feedback_context = f"""
THOUGHT (Revision): The previous draft was returned for revision.
Human feedback: "{state['human_feedback']}"
Consider what specific changes are needed."""

    response = llm.invoke(
        f"""You are a content writing agent.

THOUGHT: Review the research on "{state['topic']}".{feedback_context}
What is the central narrative? What claims are well-supported?

ACTION: Write a complete blog post synthesizing this research:
{context}

Format: Title, introduction, 3-4 body sections, conclusion."""
    )
    return {"draft_content": response.content}

# --- Node 5: Human Reviews Draft (TAPO: Pause + Observation) ---
def review_draft(state: PipelineState) -> Command[Literal["publisher", "writer_agent", "__end__"]]:
    # TAPO Pause: interrupt surfaces draft for human review
    decision = interrupt({
        "step": "review_draft",
        "tapo_phase": "PAUSE — awaiting human observation",
        "draft": state["draft_content"],
        "instruction": "Approve, edit, request revision, or reject."
    })
    # TAPO Observation: human feedback determines next cycle
    if decision["action"] == "approve":
        return Command(goto="publisher",
                      update={"final_content": state["draft_content"],
                              "human_feedback": ""})
    elif decision["action"] == "edit":
        return Command(goto="publisher",
                      update={"final_content": decision["edited_content"],
                              "human_feedback": ""})
    elif decision["action"] == "revise":
        # Observation feeds back into Writer's next Thought step
        return Command(goto="writer_agent",
                      update={"human_feedback": decision["feedback"],
                              "status": "revision_requested"})
    return Command(goto="__end__", update={"status": "rejected"})

# --- Node 6: Publisher with Final Confirmation (TAPO: Full Cycle) ---
def publisher(state: PipelineState):
    # TAPO Pause: final go/no-go before irreversible action
    confirm = interrupt({
        "step": "confirm_publish",
        "tapo_phase": "PAUSE — final human gate",
        "content_preview": state["final_content"][:500],
        "instruction": "Confirm publishing? (yes/no)"
    })
    # TAPO Observation: binary — publish or cancel
    if confirm.get("action") == "confirm":
        publish_to_platform(state["final_content"])
        return {"status": "published"}
    return {"status": "publish_cancelled"}
```

### Step 3: Wire the Graph

Connect all nodes with edges and compile with the checkpointer.

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

builder = StateGraph(PipelineState)

# Add all nodes
builder.add_node("research_agent", research_agent)
builder.add_node("review_queries", review_queries)
builder.add_node("web_search", web_search)
builder.add_node("writer_agent", writer_agent)
builder.add_node("review_draft", review_draft)
builder.add_node("publisher", publisher)

# Define the flow
builder.add_edge(START, "research_agent")
builder.add_edge("research_agent", "review_queries")
builder.add_edge("review_queries", "web_search")
builder.add_edge("web_search", "writer_agent")
builder.add_edge("writer_agent", "review_draft")
# review_draft uses Command(goto=...) for dynamic routing
builder.add_edge("publisher", END)

# Compile with checkpointer
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
```

### Step 4: Visualize the Graph

LangGraph provides built-in Mermaid diagram generation — essential for debugging complex flows.

```python
# Print Mermaid syntax (paste into mermaid.live)
print(graph.get_graph().draw_mermaid())

# Or render directly as PNG in Jupyter
from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png()))
```

### Step 5: Run with Human-in-the-Loop

The execution loop invokes the graph, checks for interrupts, collects human input, and resumes. Each interrupt represents a TAPO Pause, and each resume represents an Observation:

```python
import uuid

def run_pipeline(topic: str):
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # Start the pipeline
    result = graph.invoke({"topic": topic, "status": "started"}, config=config)

    # Loop through HITL checkpoints (each is a TAPO Pause → Observation)
    while result.get("__interrupt__"):
        interrupt_info = result["__interrupt__"][0]
        payload = interrupt_info.value

        print(f"\n--- TAPO PAUSE: {payload['step']} ---")
        print(f"Phase: {payload.get('tapo_phase', 'Awaiting human input')}")
        print(f"Details: {payload}")

        # Collect human input (TAPO Observation)
        # In production, this would be a web UI or Slack integration
        human_response = collect_human_input(payload)

        # Resume with human's decision (Observation → next Thought)
        result = graph.invoke(Command(resume=human_response), config=config)

    print(f"\nFinal status: {result.get('status')}")
    return result

# Run it
run_pipeline("The impact of AI agents on software development in 2025")
```

---

## Five Rules That Prevent Subtle Interrupt Bugs

The `interrupt()` mechanism has critical gotchas that cause silent failures if ignored. These come directly from the official LangGraph documentation and are the most common source of bugs in HITL implementations.

**First, never wrap `interrupt()` in try/except.** The function works by throwing a `GraphInterrupt` exception. Catching it with a broad `except Exception` clause swallows the interrupt entirely, and the graph will continue executing instead of pausing. Always separate interrupt calls from error-prone code.

**Second, the entire node re-executes on resume.** When `Command(resume=...)` is called, LangGraph replays the node from the beginning. The `interrupt()` call returns the resume value instead of pausing again, but all code before it runs again. This means any side effects (API calls, database writes) before the interrupt must be **idempotent** or guarded with checks. This is especially important when your TAPO Action step involves external API calls — make sure they're safe to repeat.

**Third, never reorder `interrupt()` calls within a node.** If a node contains multiple `interrupt()` calls, the runtime matches resume values to interrupts by their execution order. Changing the order between deployments will cause incorrect value matching.

**Fourth, only pass JSON-serializable values** to `interrupt()`. Complex objects, class instances, or circular references will fail silently or throw serialization errors.

**Fifth, always use the same `thread_id`** when resuming. The thread ID is the key that links the resume call to the correct checkpointed state. A different thread ID starts a fresh execution.

---

## Multi-Agent Patterns and Where HITL + TAPO Fit

LangGraph supports several multi-agent architectures. The choice of pattern determines where human review checkpoints and TAPO cycles sit most naturally.

**The supervisor pattern** places a central coordinator that delegates to specialized worker agents. HITL checkpoints work best between the supervisor's routing decisions and the worker execution. The supervisor itself follows a TAPO cycle: it *Thinks* about which agent to delegate to, *Acts* by routing, *Pauses* for human approval of the routing decision (optional), and incorporates the *Observation* to adjust delegation.

```python
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent

research_agent = create_react_agent(model=llm, tools=[tavily_search], name="researcher")
writer_agent = create_react_agent(model=llm, tools=[write_tool], name="writer")

workflow = create_supervisor(
    [research_agent, writer_agent],
    model=llm,
    prompt="You manage a research-and-writing team."
)
app = workflow.compile(checkpointer=MemorySaver())
```

**The sequential pattern** chains agents in a fixed pipeline — exactly what our project uses. HITL checkpoints slot naturally between pipeline stages, and each agent-to-HITL-to-next-agent transition is one complete TAPO cycle. Use `add_sequence()` for simpler graphs:

```python
builder = StateGraph(State).add_sequence([researcher, reviewer, writer, publisher])
builder.add_edge(START, "researcher")
```

**The parallel pattern** fans out to multiple agents simultaneously, then merges results. HITL checkpoints go after the fan-in merge point, where the human reviews aggregated results. Each parallel agent runs its own Thought → Action cycle independently, and the human's Observation at the merge point applies to the combined output.

```python
from langgraph.types import Send

def fan_out_to_searches(state):
    return [Send("web_search", {"query": q}) for q in state["search_queries"]]

builder.add_conditional_edges("review_queries", fan_out_to_searches, ["web_search"])
```

**The hierarchical pattern** nests supervisors — team leads managing workers, coordinated by a top-level supervisor. HITL gates can sit at any level: team-level for operational decisions, top-level for strategic ones. TAPO cycles nest correspondingly — each level has its own Thought/Action scope with human Observations feeding up or down the hierarchy.

---

## Putting It Into Production

Moving from `MemorySaver` to a production setup requires three changes: swapping the checkpointer, adding proper error handling around interrupts, and building a UI layer for human interactions.

**For persistence**, replace `MemorySaver` with `PostgresSaver` for durable state across restarts:

```python
from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = PostgresSaver(connection_string="postgresql://user:pass@host/db")
graph = builder.compile(checkpointer=checkpointer)
```

**For the UI layer**, the interrupt payload can drive a web interface. Each interrupt surfaces structured data (including the `tapo_phase` label) that a frontend can render as a form, approval button, or text editor. The backend waits for the human response, then calls `graph.invoke(Command(resume=response), config)`. FastAPI integration is straightforward — one endpoint starts the graph, another resumes it with the same thread ID.

**For state inspection**, LangGraph provides `get_state()` and `get_state_history()` to examine the graph at any point:

```python
state = graph.get_state(config)
print(state.values)   # Current state dictionary
print(state.next)     # Tuple of next node(s) to execute
```

**For the v2 API** (LangGraph ≥ 1.1), use `version="v2"` for cleaner interrupt access via `result.interrupts` instead of `result["__interrupt__"]`.

---

## Incorporating Observability

A multi-agent HITL system has moving parts that are invisible without proper instrumentation: LLM latency per node, token costs per agent, human review dwell time, revision loop frequency, and end-to-end pipeline throughput. Observability turns these into measurable signals you can act on. This section covers three layers — LangSmith (native tracing), OpenTelemetry (vendor-neutral telemetry), and custom HITL/TAPO metrics — and shows how to wire them into the content pipeline.

#### Diagram: Observability Architecture

```mermaid
graph TD
    subgraph APP["LangGraph Pipeline"]
        direction TB
        RA["Research Agent"] --> H1["HITL #1"]
        H1 --> WS["Web Search"]
        WS --> WA["Writer Agent"]
        WA --> H2["HITL #2"]
        H2 --> PUB["Publisher"]
        PUB --> H3["HITL #3"]
    end

    subgraph INSTRUMENTATION["Instrumentation Layer"]
        direction TB
        LS["LangSmith Tracer<br/><i>Auto-traces LLM calls,<br/>tool invocations, state</i>"]
        OT["OpenTelemetry SDK<br/><i>Custom spans, metrics,<br/>OTLP export</i>"]
        CL["Custom Logger<br/><i>TAPO phase tracking,<br/>HITL dwell time</i>"]
    end

    subgraph BACKENDS["Observability Backends"]
        direction LR
        LSP["LangSmith Platform<br/><i>Trace explorer, evals,<br/>cost dashboard</i>"]
        GR["Grafana / Datadog<br/><i>Dashboards, alerts,<br/>SLO monitoring</i>"]
        LOG["Structured Logs<br/><i>ELK / CloudWatch<br/>audit trail</i>"]
    end

    APP -->|"callbacks"| LS
    APP -->|"spans + metrics"| OT
    APP -->|"structured events"| CL

    LS -->|"native SDK"| LSP
    OT -->|"OTLP gRPC"| GR
    OT -->|"OTLP fan-out"| LSP
    CL -->|"JSON logs"| LOG

    style RA fill:#EEEDFE,stroke:#534AB7,color:#3C3489
    style WA fill:#EEEDFE,stroke:#534AB7,color:#3C3489
    style PUB fill:#EEEDFE,stroke:#534AB7,color:#3C3489
    style WS fill:#E1F5EE,stroke:#0F6E56,color:#085041
    style H1 fill:#FAECE7,stroke:#993C1D,color:#712B13
    style H2 fill:#FAECE7,stroke:#993C1D,color:#712B13
    style H3 fill:#FAECE7,stroke:#993C1D,color:#712B13
    style LS fill:#E6F1FB,stroke:#185FA5,color:#0C447C
    style OT fill:#E6F1FB,stroke:#185FA5,color:#0C447C
    style CL fill:#E6F1FB,stroke:#185FA5,color:#0C447C
    style LSP fill:#FAEEDA,stroke:#854F0B,color:#633806
    style GR fill:#FAEEDA,stroke:#854F0B,color:#633806
    style LOG fill:#FAEEDA,stroke:#854F0B,color:#633806
    style APP fill:transparent,stroke:#B4B2A9,stroke-dasharray:5 5
    style INSTRUMENTATION fill:transparent,stroke:#B4B2A9,stroke-dasharray:5 5
    style BACKENDS fill:transparent,stroke:#B4B2A9,stroke-dasharray:5 5
```

### Layer 1: LangSmith — Zero-Config Native Tracing

LangSmith is the path of least resistance. Set two environment variables and every LLM call, tool invocation, and state transition in your LangGraph pipeline gets traced automatically — no code changes required.

**Setup (three lines):**

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY="ls-..."
export LANGCHAIN_PROJECT="content-pipeline-hitl"
```

That's it. Every `graph.invoke()` now produces a hierarchical trace in the LangSmith dashboard showing each node's execution time, token usage, input/output state, and the exact prompts sent to the LLM. For HITL nodes, the trace shows where the graph paused at `interrupt()` and what `Command(resume=value)` was received.

**What you get out of the box:**

- **Trace explorer** — drill into any pipeline run to see the full execution waterfall: which agent ran, what it called, how long it took, and what tokens it consumed.
- **Cost attribution** — token usage broken down by node, so you can see if the Writer Agent is 5× more expensive than the Research Agent.
- **Latency breakdown** — P50/P99 latency per node and for the full pipeline.
- **Feedback scores** — attach human quality scores to runs for systematic evaluation.
- **Run comparisons** — diff two pipeline runs side by side to see what changed.

**Adding custom metadata to traces:**

You can enrich traces with TAPO phase labels and HITL metadata by passing metadata through the config:

```python
from langchain_core.runnables import RunnableConfig

config = RunnableConfig(
    configurable={"thread_id": thread_id},
    metadata={
        "pipeline": "content-research-publish",
        "topic": topic,
        "tapo_checkpoint": "hitl_1_query_review"
    },
    tags=["production", "hitl-enabled"]
)

result = graph.invoke({"topic": topic}, config=config)
```

**Sensitive data masking:**

If your pipeline handles PII or confidential content, use LangSmith's anonymizer to redact before traces leave your environment:

```python
from langchain_core.tracers.langchain import LangChainTracer
from langsmith import Client
from langsmith.anonymizer import create_anonymizer

anonymizer = create_anonymizer([
    {"pattern": r"\b\d{3}-?\d{2}-?\d{4}\b", "replace": "<ssn>"},
    {"pattern": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z]{2,}\b",
     "replace": "<email>"}
])

tracer_client = Client(anonymizer=anonymizer)
tracer = LangChainTracer(client=tracer_client)
graph = builder.compile(checkpointer=checkpointer).with_config(
    {"callbacks": [tracer]}
)
```

### Layer 2: OpenTelemetry — Vendor-Neutral Production Telemetry

LangSmith is excellent for LLM-specific debugging, but production systems need telemetry that flows into your existing observability stack (Datadog, Grafana, New Relic, etc.). OpenTelemetry (OTel) provides this — and LangSmith now supports full end-to-end OTel integration so you can fan out traces to multiple backends.

**Option A: OTel-native export from LangSmith SDK**

The simplest path if you're already using LangSmith — enable the OTel bridge and route traces to both LangSmith and your APM:

```bash
pip install "langsmith[otel]"
export LANGSMITH_OTEL_ENABLED=true
export LANGSMITH_API_KEY="ls-..."
```

Then configure an OpenTelemetry Collector to fan out:

```yaml
# otel-collector-config.yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: "0.0.0.0:4317"

exporters:
  otlp/langsmith:
    endpoint: "https://api.smith.langchain.com"
    headers:
      x-api-key: "${LANGSMITH_API_KEY}"
  otlp/grafana:
    endpoint: "https://otlp-gateway.grafana.net/otlp"
    headers:
      Authorization: "Basic ${GRAFANA_TOKEN}"

service:
  pipelines:
    traces:
      receivers: [otlp]
      exporters: [otlp/langsmith, otlp/grafana]
```

**Option B: Direct OpenTelemetry instrumentation with OpenInference**

If you want vendor-neutral instrumentation without the LangSmith SDK dependency, use the OpenInference community instrumentor:

```bash
pip install opentelemetry-distro \
    opentelemetry-exporter-otlp \
    openinference-instrumentation-langchain
```

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from openinference.instrumentation.langchain import LangChainInstrumentor

# Configure OTel provider
provider = TracerProvider()
provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:4317"))
)
trace.set_tracer_provider(provider)

# Auto-instrument all LangChain/LangGraph operations
LangChainInstrumentor().instrument()

# Now compile and run your graph as normal — traces flow automatically
graph = builder.compile(checkpointer=checkpointer)
```

This captures LLM calls, tool invocations, and chain/graph execution as nested spans — viewable in Jaeger, Grafana Tempo, Datadog, or any OTel-compatible backend.

### Layer 3: Custom HITL and TAPO Metrics

The first two layers cover LLM performance. But in a Human-in-the-Loop system, the *human* is often the bottleneck. You need custom metrics that track what the native instrumentation misses.

**Key HITL metrics to capture:**

| Metric | What It Tells You | How to Capture |
|---|---|---|
| `hitl.dwell_time_seconds` | How long the human took to review | Timestamp diff: interrupt fired → resume received |
| `hitl.decision` | What the human chose (approve/edit/revise/reject) | Extract from `Command(resume=...)` value |
| `hitl.revision_count` | How many revision loops before approval | Counter incremented on each `revise` decision |
| `tapo.thought_tokens` | Token cost of the agent's reasoning step | Extract from LLM response metadata |
| `tapo.action_latency_seconds` | How long the agent's action took | Span duration of the node function |
| `pipeline.end_to_end_seconds` | Total wall-clock time including human delays | Timestamp diff: first invoke → final status |
| `pipeline.outcome` | Published / cancelled / rejected | Final `status` field in state |

**Implementation — custom instrumented HITL node:**

```python
import time
import structlog
from opentelemetry import trace, metrics

# Set up OTel instruments
tracer = trace.get_tracer("content-pipeline")
meter = metrics.get_meter("content-pipeline")

hitl_dwell_histogram = meter.create_histogram(
    "hitl.dwell_time_seconds",
    description="Time human spent reviewing at each checkpoint",
    unit="s"
)
hitl_decision_counter = meter.create_counter(
    "hitl.decision_total",
    description="Count of human decisions by type"
)
revision_counter = meter.create_counter(
    "hitl.revision_count",
    description="Number of revision loops triggered"
)

logger = structlog.get_logger()

def review_draft_with_observability(state: PipelineState):
    """HITL #2 with full observability instrumentation."""

    with tracer.start_as_current_span("hitl.review_draft") as span:
        span.set_attribute("tapo.phase", "PAUSE")
        span.set_attribute("hitl.checkpoint", "review_draft")
        span.set_attribute("draft.length", len(state.get("draft_content", "")))

        # Log the TAPO Pause event
        logger.info("tapo_pause",
            checkpoint="review_draft",
            draft_length=len(state.get("draft_content", "")),
            thread_id=state.get("thread_id")
        )

        # Record timestamp before interrupt
        pause_start = time.time()

        # TAPO Pause — graph halts here
        decision = interrupt({
            "step": "review_draft",
            "tapo_phase": "PAUSE",
            "draft": state["draft_content"],
            "instruction": "Approve, edit, request revision, or reject."
        })

        # TAPO Observation — human has responded
        dwell_time = time.time() - pause_start

        # Record HITL metrics
        hitl_dwell_histogram.record(dwell_time, {"checkpoint": "review_draft"})
        hitl_decision_counter.add(1, {
            "checkpoint": "review_draft",
            "decision": decision["action"]
        })

        span.set_attribute("tapo.phase", "OBSERVATION")
        span.set_attribute("hitl.dwell_time_seconds", dwell_time)
        span.set_attribute("hitl.decision", decision["action"])

        logger.info("tapo_observation",
            checkpoint="review_draft",
            decision=decision["action"],
            dwell_time_seconds=round(dwell_time, 2)
        )

        if decision["action"] == "approve":
            return Command(goto="publisher",
                          update={"final_content": state["draft_content"]})
        elif decision["action"] == "edit":
            return Command(goto="publisher",
                          update={"final_content": decision["edited_content"]})
        elif decision["action"] == "revise":
            revision_counter.add(1, {"checkpoint": "review_draft"})
            logger.info("revision_requested",
                feedback=decision.get("feedback", ""),
                revision_number=state.get("revision_count", 0) + 1
            )
            return Command(goto="writer_agent",
                          update={
                              "human_feedback": decision["feedback"],
                              "revision_count": state.get("revision_count", 0) + 1
                          })

        return Command(goto="__end__", update={"status": "rejected"})
```

### Layer 4: Structured Logging for TAPO Phase Tracking

Beyond metrics, structured logs create an audit trail that maps every pipeline run to its TAPO phases. This is invaluable for debugging ("why did the Writer Agent produce a poor draft?") and compliance ("who approved this publication and when?").

**Structured log schema per TAPO event:**

```python
import structlog
import datetime

logger = structlog.get_logger()

def log_tapo_event(phase: str, node: str, details: dict):
    """Emit a structured log for each TAPO phase transition."""
    logger.info("tapo_event",
        timestamp=datetime.datetime.utcnow().isoformat(),
        phase=phase,           # THOUGHT | ACTION | PAUSE | OBSERVATION
        node=node,             # research_agent | review_queries | etc.
        thread_id=details.get("thread_id"),
        pipeline_run_id=details.get("run_id"),
        **details
    )

# Usage inside agent nodes:

# In research_agent (before LLM call)
log_tapo_event("THOUGHT", "research_agent", {
    "topic": state["topic"],
    "has_prior_feedback": bool(state.get("human_feedback")),
    "thread_id": config["configurable"]["thread_id"]
})

# In research_agent (after LLM returns queries)
log_tapo_event("ACTION", "research_agent", {
    "queries_generated": len(queries),
    "token_usage": response.response_metadata.get("token_usage", {}),
    "thread_id": config["configurable"]["thread_id"]
})

# In review_queries (when interrupt fires)
log_tapo_event("PAUSE", "review_queries", {
    "queries_presented": state["search_queries"],
    "thread_id": config["configurable"]["thread_id"]
})

# In review_queries (when human resumes)
log_tapo_event("OBSERVATION", "review_queries", {
    "decision": decision["action"],
    "dwell_time_seconds": dwell_time,
    "queries_modified": decision["action"] == "edit",
    "thread_id": config["configurable"]["thread_id"]
})
```

**Sample log output (JSON lines):**

```json
{"event":"tapo_event","phase":"THOUGHT","node":"research_agent","topic":"AI agents 2025","has_prior_feedback":false,"thread_id":"abc-123","timestamp":"2025-12-15T10:30:01Z"}
{"event":"tapo_event","phase":"ACTION","node":"research_agent","queries_generated":3,"token_usage":{"input":245,"output":89},"thread_id":"abc-123","timestamp":"2025-12-15T10:30:03Z"}
{"event":"tapo_event","phase":"PAUSE","node":"review_queries","queries_presented":["AI agent frameworks 2025","..."],"thread_id":"abc-123","timestamp":"2025-12-15T10:30:03Z"}
{"event":"tapo_event","phase":"OBSERVATION","node":"review_queries","decision":"edit","dwell_time_seconds":47.3,"queries_modified":true,"thread_id":"abc-123","timestamp":"2025-12-15T10:30:50Z"}
```

These logs can be shipped to ELK, CloudWatch, or any log aggregator and queried by `thread_id` to reconstruct the full TAPO trace of any pipeline run.

### Choosing Your Observability Stack

| Scenario | Recommended Stack | Why |
|---|---|---|
| **Dev / prototyping** | LangSmith only | Zero config, instant trace visibility, free tier |
| **Single-team production** | LangSmith + structured logs | LangSmith for LLM debugging, logs for HITL audit trail |
| **Enterprise / multi-team** | LangSmith + OpenTelemetry + Grafana/Datadog | OTel fan-out to existing APM, LangSmith for LLM-specific drill-down |
| **Regulated / compliance** | All three layers + log archival | Full TAPO audit trail, anonymized traces, immutable log storage |

### Building a HITL Observability Dashboard

The custom metrics above feed into a dashboard that answers the questions a pipeline operator actually cares about:

**Top row — real-time pipeline health:**
- Active pipeline runs (count of threads with status != terminal)
- End-to-end P50 / P95 latency (including human dwell time)
- Approval rate across all HITL checkpoints (approvals / total decisions)

**Middle row — human reviewer performance:**
- Dwell time distribution by checkpoint (histogram)
- Decision breakdown by checkpoint (stacked bar: approve / edit / revise / reject)
- Revision loops per run (are certain topics consistently requiring revisions?)

**Bottom row — agent cost and quality:**
- Token usage by agent node (Research vs Writer vs Publisher)
- Cost per pipeline run (tokens × model price)
- Revision correlation (do certain topics or query types lead to more revisions?)

**Alerting rules to set up:**
- `hitl.dwell_time_seconds > 3600` on any checkpoint → human reviewer may be stuck
- `hitl.revision_count > 3` on a single run → possible prompt quality issue
- `pipeline.outcome == "rejected"` rate > 20% over 1 hour → systemic problem
- LLM error rate > 5% → model API issues
- Token usage per run > 2× rolling average → prompt regression or runaway generation

---

## Quick Reference: TAPO ↔ LangGraph Cheat Sheet

| TAPO Step | What Happens | LangGraph Mechanism | Where in Code |
|---|---|---|---|
| **Thought** | Agent reasons about task, plans approach | System prompt with chain-of-thought instructions | Inside `llm.invoke()` call |
| **Action** | Agent executes plan, produces output | Node function body, returns state update | `return {"field": value}` |
| **Pause** | Graph halts, human reviews output | `interrupt(payload)` | Inside HITL node |
| **Observation** | Human feedback enters the system | `Command(resume=value)` | Caller's `graph.invoke()` |

---

## Conclusion

LangGraph's `interrupt()` and `Command(resume=...)` pattern represents a significant simplification over the older `interrupt_before`/`interrupt_after` approach. When combined with the TAPO prompting cycle, the result is a system where agent reasoning is transparent, human oversight is structurally guaranteed, and feedback loops are explicit in both the prompt design and the graph topology. Adding observability across all three layers — LangSmith for LLM-specific tracing, OpenTelemetry for vendor-neutral production telemetry, and custom HITL metrics for human reviewer performance — ensures you can diagnose issues at any point in the TAPO cycle: a slow Thought (prompt regression), a failed Action (tool error), a stalled Pause (reviewer bottleneck), or a weak Observation (insufficient feedback signal).

The key architectural insight is that **interrupts are dynamic and composable** — they can be conditional, nested inside loops for validation, or embedded within tool functions. The TAPO cycle gives you a mental model for *where* to place them: always between Action and the next Thought, so the human's Observation directly informs what the agent thinks about next. Observability gives you a mental model for *how well* they're working: dwell time tells you if the Pause is too frequent or too ambiguous, revision counts tell you if the Observation is feeding back effectively, and token costs tell you if the Thought/Action prompts need tightening.

Start with `SqliteSaver` + LangSmith for development (this project ships with `SqliteSaver` as the default), then graduate to `PostgresSaver` + OpenTelemetry + structured logging as you move to production. The TAPO prompting structure and the observability instrumentation both scale with you — they work the same way whether you have one agent or twenty.
