"""
api.py — FastAPI Server for Production HITL
---------------------------------------------
Exposes the pipeline as a REST API so a web frontend (or Slack bot, etc.)
can drive the HITL loop instead of a terminal.

Two core endpoints:
  POST /pipeline/start          → starts a new pipeline run, returns thread_id
                                   and the first interrupt payload (if any)
  POST /pipeline/resume         → resumes a paused pipeline with human's decision
  GET  /pipeline/{thread_id}    → inspect current state of any pipeline run

Flow from a frontend perspective:
  1. POST /start with {"topic": "..."} → get back thread_id + interrupt payload
  2. Show the payload in your UI (queries to approve, draft to review, etc.)
  3. Human makes a decision → POST /resume with thread_id + their response
  4. Repeat step 2-3 until response has no "interrupt" field (pipeline done)

This is stateless on the API side — all state lives in the LangGraph checkpointer.
"""

import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Optional

from langgraph.types import Command
from graph import graph

app = FastAPI(
    title="Content Pipeline API",
    description="Multi-agent HITL content research and publishing pipeline",
    version="1.0.0"
)


# --- Request / Response Models ---

class StartRequest(BaseModel):
    topic: str


class ResumeRequest(BaseModel):
    thread_id: str
    response: dict  # The human's JSON decision, e.g. {"action": "approve"}


class PipelineResponse(BaseModel):
    thread_id: str
    status: str
    interrupt: Optional[dict] = None   # Present when pipeline is paused
    final_status: Optional[str] = None  # Present when pipeline is complete


# --- Helper ---

def _extract_interrupt(result: dict) -> Optional[dict]:
    """Pulls the interrupt payload out of the graph result, if present."""
    interrupts = result.get("__interrupt__")
    if interrupts:
        return interrupts[0].value
    return None


# --- Endpoints ---

@app.post("/pipeline/start", response_model=PipelineResponse)
def start_pipeline(req: StartRequest):
    """
    Starts a new pipeline run.
    Returns immediately with the first interrupt payload (HITL #1: query review).
    The client should display the payload and collect the human's decision.
    """
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    result = graph.invoke(
        {"topic": req.topic, "status": "started", "human_feedback": ""},
        config=config
    )

    interrupt_payload = _extract_interrupt(result)

    return PipelineResponse(
        thread_id=thread_id,
        status="paused" if interrupt_payload else "complete",
        interrupt=interrupt_payload,
        final_status=result.get("status") if not interrupt_payload else None
    )


@app.post("/pipeline/resume", response_model=PipelineResponse)
def resume_pipeline(req: ResumeRequest):
    """
    Resumes a paused pipeline with the human's decision.
    The thread_id must match the one returned by /start (or a previous /resume).
    Returns the next interrupt payload, or final_status if the pipeline is done.
    """
    config = {"configurable": {"thread_id": req.thread_id}}

    # Verify the thread exists
    try:
        state = graph.get_state(config)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Thread '{req.thread_id}' not found.")

    if not state.next:
        raise HTTPException(status_code=400, detail="Pipeline is already complete.")

    # Resume with human's decision (TAPO Observation)
    result = graph.invoke(Command(resume=req.response), config=config)

    interrupt_payload = _extract_interrupt(result)

    return PipelineResponse(
        thread_id=req.thread_id,
        status="paused" if interrupt_payload else "complete",
        interrupt=interrupt_payload,
        final_status=result.get("status") if not interrupt_payload else None
    )


@app.get("/pipeline/{thread_id}", response_model=dict)
def get_pipeline_state(thread_id: str):
    """
    Returns the current state of a pipeline run.
    Useful for debugging or building a status dashboard.
    """
    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = graph.get_state(config)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' not found.")

    return {
        "thread_id": thread_id,
        "values": state.values,
        "next_nodes": list(state.next),
        "is_complete": len(state.next) == 0
    }


@app.get("/health")
def health():
    return {"status": "ok"}
