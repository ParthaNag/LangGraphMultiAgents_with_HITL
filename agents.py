"""
agents.py — All 6 Node Functions
----------------------------------
Each function is a LangGraph node. Nodes receive the full PipelineState
and return only the fields they modify (partial state updates).

The 6 nodes follow the pipeline in order:
  1. research_agent     → TAPO: Thought + Action  (generates search queries)
  2. review_queries     → TAPO: Pause + Observation (human approves/edits queries)
  3. web_search_node    → executes the approved queries
  4. writer_agent       → TAPO: Thought + Action  (writes the blog post)
  5. review_draft       → TAPO: Pause + Observation (human approves/edits/revises)
  6. publisher          → TAPO: Pause + Observation (final confirm before publish)

TAPO recap per node:
  Thought  = system prompt reasoning instructions passed to the LLM
  Action   = the node function body that returns a state update
  Pause    = interrupt(payload) — halts graph, checkpoints state
  Observation = Command(resume=value) — human's response fed back in
"""

from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel
from langgraph.types import interrupt, Command

from state import PipelineState
from tools import web_search, publish_to_platform
from llm_config import get_llm


class _SearchQueries(BaseModel):
    queries: list[str]

load_dotenv()


# ---------------------------------------------------------------------------
# NODE 1: Research Agent
# TAPO Phase: Thought + Action
# What it does: Takes the user's topic and generates 3 focused search queries.
# The TAPO Thought is embedded in the system prompt — the LLM reasons about
# what a domain expert would search for before generating queries.
# ---------------------------------------------------------------------------
def research_agent(state: PipelineState) -> dict:
    # If there's prior human feedback (e.g. queries were rejected and restarted),
    # inject it into the Thought step so the agent learns from it
    feedback_context = ""
    if state.get("human_feedback"):
        feedback_context = f"\n\nPrevious human feedback to incorporate: \"{state['human_feedback']}\""

    prompt = f"""You are a research planning agent.

THOUGHT: Analyze the topic "{state['topic']}".{feedback_context}
Consider what a domain expert would search for and what gaps might exist in obvious queries.

ACTION: Generate exactly 3 focused, complementary search queries for this topic.
Each query should target a different angle.
"""

    # with_structured_output enforces the schema at the LLM level — no regex parsing needed
    print("[research_agent] Generating queries...")
    structured_llm = get_llm().with_structured_output(_SearchQueries)
    result = structured_llm.invoke(prompt)
    queries = result.queries

    print(f"\n[research_agent] Generated {len(queries)} queries for topic: '{state['topic']}'")
    return {"search_queries": queries, "status": "queries_generated"}


# ---------------------------------------------------------------------------
# NODE 2: Review Queries (HITL Checkpoint #1)
# TAPO Phase: Pause + Observation
# What it does: Pauses the graph and surfaces the queries to the human.
# The human can approve, edit (provide new queries), or reject entirely.
# interrupt() is the pause — Command(resume=...) is the observation.
# ---------------------------------------------------------------------------
def review_queries(state: PipelineState) -> dict | Command:
    # TAPO PAUSE — graph halts here, state is checkpointed
    decision = interrupt({
        "step": "review_queries",
        "tapo_phase": "PAUSE — awaiting human observation",
        "suggested_queries": state["search_queries"],
        "instruction": (
            "Review the search queries below.\n"
            "Respond with one of:\n"
            "  {\"action\": \"approve\"}\n"
            "  {\"action\": \"edit\", \"queries\": [\"q1\", \"q2\", \"q3\"], \"reason\": \"why\"}\n"
            "  {\"action\": \"reject\"}"
        )
    })

    # TAPO OBSERVATION — human's response is now in `decision`
    action = decision.get("action", "reject")

    if action == "approve":
        print("[review_queries] Human approved queries.")
        return {
            "search_queries": state["search_queries"],
            "status": "queries_approved",
            "human_feedback": ""
        }

    elif action == "edit":
        edited = decision.get("queries", state["search_queries"])
        reason = decision.get("reason", "")
        print(f"[review_queries] Human edited queries. Reason: {reason}")
        return {
            "search_queries": edited,
            "status": "queries_edited",
            "human_feedback": reason
        }

    # Reject — route to end
    print("[review_queries] Human rejected queries. Ending pipeline.")
    return Command(goto="__end__", update={"status": "cancelled_at_queries"})


# ---------------------------------------------------------------------------
# NODE 3: Web Search
# TAPO Phase: N/A (pure tool execution, no LLM)
# What it does: Runs each approved query through Tavily and collects results.
# Results are appended to research_results (operator.add reducer in state.py).
# ---------------------------------------------------------------------------
def web_search_node(state: PipelineState) -> dict:
    all_results = []
    for query in state["search_queries"]:
        print(f"[web_search] Searching: '{query}'")
        results = web_search(query, max_results=3)
        all_results.extend(results)

    print(f"[web_search] Collected {len(all_results)} result snippets.")
    return {"research_results": all_results, "status": "research_complete"}


# ---------------------------------------------------------------------------
# NODE 4: Writer Agent
# TAPO Phase: Thought + Action
# What it does: Takes all research results and writes a full blog post.
# If human_feedback is set (revision loop), the Thought step explicitly
# references what was wrong with the previous draft — this is the nested
# TAPO cycle described in the guide.
# ---------------------------------------------------------------------------
def writer_agent(state: PipelineState) -> dict:
    context = "\n\n---\n\n".join(state.get("research_results", []))

    # Nested TAPO: if this is a revision, the Thought step changes
    if state.get("human_feedback"):
        thought_section = f"""THOUGHT (Revision Round):
The previous draft was returned for revision.
Human feedback: "{state['human_feedback']}"
Consider exactly what changes are needed and why the previous version fell short.
Preserve the strengths of the original while addressing the feedback directly."""
    else:
        thought_section = f"""THOUGHT:
Review the research results on "{state['topic']}".
Consider:
- What is the central narrative these findings support?
- What claims are well-supported vs. speculative?
- What structure will best serve the reader?"""

    # Trim research context to keep costs reasonable (Gemini has 1M ctx window)
    trimmed_context = context[:8000] if len(context) > 8000 else context

    prompt = f"""You are a content writing agent.

{thought_section}

ACTION: Write a blog post (300-400 words) synthesizing this research.
Format: Title, short intro, 2-3 sections with headers, conclusion.
Tone: Clear and direct.

Research:
{trimmed_context}
"""

    print("[writer_agent] Generating draft...")
    response = get_llm().invoke(prompt)
    print(f"[writer_agent] Draft written ({len(response.content)} chars).")
    return {
        "draft_content": response.content,
        "status": "draft_written",
        "human_feedback": ""  # clear feedback after incorporating it
    }


# ---------------------------------------------------------------------------
# NODE 5: Review Draft (HITL Checkpoint #2)
# TAPO Phase: Pause + Observation
# What it does: Pauses for human review of the full draft.
# Three paths:
#   approve → move to publisher
#   edit    → human provides edited text directly, move to publisher
#   revise  → send back to writer_agent with feedback (revision loop)
#   reject  → end pipeline
# The Command(goto=...) enables dynamic routing from inside the node.
# ---------------------------------------------------------------------------
def review_draft(state: PipelineState) -> Command[Literal["publisher", "writer_agent", "__end__"]]:
    # TAPO PAUSE
    decision = interrupt({
        "step": "review_draft",
        "tapo_phase": "PAUSE — awaiting human observation",
        "draft": state["draft_content"],
        "instruction": (
            "Review the draft below.\n"
            "Respond with one of:\n"
            "  {\"action\": \"approve\"}\n"
            "  {\"action\": \"edit\", \"edited_content\": \"<your full edited text>\"}\n"
            "  {\"action\": \"revise\", \"feedback\": \"what needs to change\"}\n"
            "  {\"action\": \"reject\"}"
        )
    })

    # TAPO OBSERVATION
    action = decision.get("action", "reject")

    if action == "approve":
        print("[review_draft] Human approved draft.")
        return Command(
            goto="publisher",
            update={
                "final_content": state["draft_content"],
                "status": "draft_approved",
                "human_feedback": ""
            }
        )

    elif action == "edit":
        # Human edited the text directly — use their version
        edited = decision.get("edited_content", state["draft_content"])
        print("[review_draft] Human edited draft directly.")
        return Command(
            goto="publisher",
            update={
                "final_content": edited,
                "status": "draft_edited",
                "human_feedback": ""
            }
        )

    elif action == "revise":
        # Send back to writer with feedback — this triggers the nested TAPO cycle
        feedback = decision.get("feedback", "Please improve the draft.")
        print(f"[review_draft] Revision requested: '{feedback}'")
        return Command(
            goto="writer_agent",
            update={
                "human_feedback": feedback,
                "status": "revision_requested"
            }
        )

    # Reject
    print("[review_draft] Human rejected draft. Ending pipeline.")
    return Command(goto="__end__", update={"status": "cancelled_at_draft"})


# ---------------------------------------------------------------------------
# NODE 6: Publisher
# TAPO Phase: Full Cycle (Thought + Action + Pause + Observation)
# What it does: Formats the final content, then pauses for a final go/no-go
# before calling publish_to_platform (the irreversible external action).
# This is the lightest HITL checkpoint but the most consequential.
# ---------------------------------------------------------------------------
def publisher(state: PipelineState) -> dict:
    # TAPO THOUGHT + ACTION: prepare a publish summary for the human to review
    content = state.get("final_content", state.get("draft_content", ""))
    preview = content[:500] + "..." if len(content) > 500 else content

    # TAPO PAUSE — final gate before irreversible action
    confirm = interrupt({
        "step": "confirm_publish",
        "tapo_phase": "PAUSE — final human gate before publish",
        "content_preview": preview,
        "word_count": len(content.split()),
        "instruction": (
            "Final confirmation before publishing.\n"
            "Respond with:\n"
            "  {\"action\": \"confirm\"}\n"
            "  {\"action\": \"cancel\"}"
        )
    })

    # TAPO OBSERVATION — binary decision
    if confirm.get("action") == "confirm":
        result = publish_to_platform(content)
        print(f"[publisher] Published! URL: {result.get('url')}")
        return {"status": "published"}

    print("[publisher] Publish cancelled by human.")
    return {"status": "publish_cancelled"}
