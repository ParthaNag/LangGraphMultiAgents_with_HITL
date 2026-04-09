"""
main.py — CLI Runner with Human-in-the-Loop
---------------------------------------------
This is the entry point for running the pipeline locally in your terminal.
It handles the full HITL loop: start → interrupt → collect input → resume → repeat.

How it works:
1. Invoke the graph with the topic
2. Check if the result contains an interrupt (TAPO Pause)
3. Print the interrupt payload so the human can see what needs review
4. Collect the human's response from stdin
5. Resume the graph with Command(resume=response)
6. Repeat until no more interrupts (pipeline complete)

This is the simplest possible HITL runner — no web UI, just terminal I/O.
For a web-based UI, see api.py.
"""

import uuid
import json
from langgraph.types import Command
from graph import graph


def collect_human_input(payload: dict) -> dict:
    """
    Prompts the human in the terminal and returns their parsed JSON response.
    In production this would be replaced by a web form, Slack bot, etc.
    """
    step = payload.get("step", "unknown")
    print(f"\n{'='*60}")
    print(f"  HITL CHECKPOINT: {step}")
    print(f"  {payload.get('tapo_phase', '')}")
    print(f"{'='*60}")

    # Show the relevant content for this checkpoint
    if step == "review_queries":
        print("\nSuggested search queries:")
        for i, q in enumerate(payload.get("suggested_queries", []), 1):
            print(f"  {i}. {q}")

    elif step == "review_draft":
        print("\nDraft content:")
        print("-" * 40)
        draft = payload.get("draft", "")
        # Show first 800 chars in terminal to keep it readable
        print(draft[:800] + "\n...[truncated]" if len(draft) > 800 else draft)
        print("-" * 40)

    elif step == "confirm_publish":
        print(f"\nContent preview ({payload.get('word_count', '?')} words):")
        print("-" * 40)
        print(payload.get("content_preview", ""))
        print("-" * 40)

    print(f"\nInstruction: {payload.get('instruction', '')}")
    print()

    # Keep prompting until we get valid JSON
    while True:
        raw = input("Your response (JSON): ").strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            print("  Invalid JSON. Try again. Example: {\"action\": \"approve\"}")


def run_pipeline(topic: str):
    """
    Runs the full Content Research & Publishing Pipeline for a given topic.
    Loops through all HITL checkpoints until the pipeline completes.
    """
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print(f"\nStarting pipeline for topic: '{topic}'")
    print(f"Thread ID: {thread_id}\n")

    # Initial invocation — runs until the first interrupt()
    result = graph.invoke(
        {"topic": topic, "status": "started", "human_feedback": ""},
        config=config
    )

    # HITL loop — each iteration is one TAPO Pause → Observation cycle
    while result.get("__interrupt__"):
        interrupt_data = result["__interrupt__"][0]
        payload = interrupt_data.value

        # Collect human input (TAPO Observation)
        human_response = collect_human_input(payload)

        # Resume graph with human's decision
        result = graph.invoke(Command(resume=human_response), config=config)

    # Pipeline complete
    final_status = result.get("status", "unknown")
    print(f"\n{'='*60}")
    print(f"  Pipeline complete. Final status: {final_status}")
    print(f"{'='*60}\n")
    return result


if __name__ == "__main__":
    import sys
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "The impact of AI agents on software development in 2025"
    run_pipeline(topic)
