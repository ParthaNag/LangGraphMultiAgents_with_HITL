"""
graph.py — Graph Assembly
--------------------------
This file wires all nodes together into a compiled LangGraph StateGraph.
Think of it as the "blueprint" — it defines the shape of the pipeline
but doesn't run anything. main.py and api.py import `graph` from here.

Key decisions made here:
- Node registration (add_node)
- Edge definitions (add_edge) — fixed sequential flow
- Dynamic routing is handled inside nodes via Command(goto=...) so
  review_draft and review_queries don't need conditional edges here
- Checkpointer attachment — SqliteSaver by default, injectable for prod
"""

import sqlite3
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from state import PipelineState
from agents import (
    research_agent,
    review_queries,
    web_search_node,
    writer_agent,
    review_draft,
    publisher,
)


def build_graph(checkpointer=None):
    """
    Builds and compiles the pipeline graph.
    Defaults to SqliteSaver (checkpoints.db) so state survives restarts.
    Pass a custom checkpointer to override (e.g. PostgresSaver for cloud).
    """
    builder = StateGraph(PipelineState)

    # --- Register all nodes ---
    builder.add_node("research_agent", research_agent)
    builder.add_node("review_queries", review_queries)
    builder.add_node("web_search_node", web_search_node)
    builder.add_node("writer_agent", writer_agent)
    builder.add_node("review_draft", review_draft)
    builder.add_node("publisher", publisher)

    # --- Define the fixed sequential flow ---
    # START → research → HITL#1 → search → write → HITL#2 → publish → HITL#3 → END
    builder.add_edge(START, "research_agent")
    builder.add_edge("research_agent", "review_queries")
    builder.add_edge("review_queries", "web_search_node")
    builder.add_edge("web_search_node", "writer_agent")
    builder.add_edge("writer_agent", "review_draft")
    # review_draft uses Command(goto=...) internally — no edge needed from it
    # publisher uses Command(resume=...) internally — just needs an END edge
    builder.add_edge("publisher", END)

    # --- Compile with checkpointer ---
    if checkpointer is not None:
        cp = checkpointer
    else:
        # SqliteSaver survives process restarts; MemorySaver is the dev fallback
        conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
        cp = SqliteSaver(conn)
    return builder.compile(checkpointer=cp)


# Module-level graph instance — imported by main.py and api.py
graph = build_graph()
