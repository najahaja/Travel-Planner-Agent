"""
=============================================================
  LangGraph Travel Planner Agent
=============================================================
  FEATURES:
    [1] Real Tool Integration  → Tavily web search in Executor
    [2] Human-in-the-Loop     → User approves plan before exec
    [3] Persistence / Memory  → MemorySaver + thread_id
    [4] Observability         → LangSmith tracing (auto if key set)
    [5] Evaluation ready      → run_headless() for eval_agent.py

  Architecture:
    START
      │
      ▼
   ┌──────────┐
   │  Planner │  ← creates 3-step plan
   └──────────┘
        │
        ▼
   ╔══════════════════════╗
   ║  HUMAN-IN-THE-LOOP   ║  ← agent PAUSES here, shows plan,
   ║  (interrupt_before   ║    waits for your "approve" / "reject"
   ║   = ["Executor"])    ║
   ╚══════════════════════╝
        │ approved
        ▼
   ┌──────────┐   Tavily   ┌──────────────────┐
   │ Executor │ ─────────▶ │  Web Search API  │
   └──────────┘            └──────────────────┘
        │
        ▼
   ┌──────────────────┐
   │  Check-Progress  │  ← conditional edge
   └──────────────────┘
        │
   ┌────┴────┐
 [done]  [more steps]
   │         │
  END    back to Executor

  Persistence:
    MemorySaver stores every state snapshot under a thread_id.
    You can resume a session any time by re-using the same thread_id.
=============================================================
"""

import os
import sys
import uuid

# ── Force UTF-8 on Windows ────────────────────────────────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from typing import TypedDict, List, Optional
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

# ── Tavily import (graceful fallback if key is missing) ───────────────────────
try:
    from langchain_community.tools.tavily_search import TavilySearchResults
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False

# ── Load environment variables ────────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")
TAVILY_API_KEY    = os.getenv("TAVILY_API_KEY", "")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")

if not GROQ_API_KEY:
    raise EnvironmentError("GROQ_API_KEY not found in .env")

# ── [UPGRADE 4] LangSmith Observability / Tracing ────────────────────────────
# When LANGCHAIN_API_KEY is present in .env, every LLM call, tool invocation,
# and state transition is automatically traced and visible in the LangSmith UI
# at https://smith.langchain.com  — zero extra code needed beyond these env vars.
if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"]  = "true"
    os.environ["LANGCHAIN_API_KEY"]     = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"]     = os.getenv(
        "LANGCHAIN_PROJECT", "travel-planner-agent"
    )
    print("  [INFO] LangSmith tracing: ENABLED")
    print(f"  [INFO] Project: {os.environ['LANGCHAIN_PROJECT']}")
    print("  [INFO] View traces at https://smith.langchain.com")
else:
    print("  [INFO] LangSmith tracing: DISABLED (no LANGCHAIN_API_KEY in .env)")

# ── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
    temperature=0.7,
)

# ── Tavily search tool (if API key is present) ────────────────────────────────
search_tool: Optional[TavilySearchResults] = None
if TAVILY_AVAILABLE and TAVILY_API_KEY:
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY          # tool reads from env
    search_tool = TavilySearchResults(max_results=3)
    print("  [INFO] Tavily search: ENABLED (real web results)")
else:
    print("  [INFO] Tavily search: DISABLED (no TAVILY_API_KEY — using LLM fallback)")


# ══════════════════════════════════════════════════════════════════════════════
#  STATE  (TypedDict)
# ══════════════════════════════════════════════════════════════════════════════
class TravelPlannerState(TypedDict):
    """Shared state for the entire graph — persisted by MemorySaver."""
    destination:    str            # Destination entered by user
    num_days:       int            # Number of days requested for the trip
    plan:           List[str]      # High-level plan step titles
    executed_steps: List[str]      # Detailed results per step
    current_step:   int            # Step index currently being processed
    feedback:       str            # Planner's review message
    is_complete:    bool           # True once all 3 steps are done
    human_approved: bool           # True once user clicks "approve"
    search_results: List[str]      # Raw Tavily snippets for current step


# ══════════════════════════════════════════════════════════════════════════════
#  NODE 1 — PLANNER
# ══════════════════════════════════════════════════════════════════════════════
def planner_node(state: TravelPlannerState) -> TravelPlannerState:
    """
    First call  → generates the 3-step plan and sets human_approved=False
                   (the graph will PAUSE before Executor until user approves).
    Later calls → reviews execution quality and marks final feedback.
    """
    destination   = state["destination"]
    existing_plan = state.get("plan", [])
    executed      = state.get("executed_steps", [])

    if not existing_plan:
        num_days = state.get("num_days", 3)
        print("\n" + "─" * 62)
        print(f"  [PLANNER]  Generating {num_days}-step itinerary plan...")
        print("─" * 62)

        system_prompt = (
            "You are an expert travel planner. "
            f"Create exactly {num_days} high-level itinerary steps for the destination. "
            "Return ONLY a numbered list:\n"
            "1. Step one\n2. Step two\n...\n"
            "Each step should be one crisp sentence. No extra text."
        )
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Plan a {num_days}-day trip to {destination}."),
        ])
        raw  = response.content.strip()
        plan = [
            line.split(". ", 1)[1].strip()
            for line in raw.splitlines()
            if line.strip() and line[0].isdigit()
        ][:num_days]

        print("\n  Proposed 3-step plan:")
        for i, s in enumerate(plan, 1):
            print(f"    {i}. {s}")
        print()

        return {
            **state,
            "plan":           plan,
            "executed_steps": [],
            "current_step":   0,
            "is_complete":    False,
            "human_approved": False,   # ← triggers Human-in-the-Loop pause
            "search_results": [],
            "feedback":       "",
        }

    else:
        # Review pass after all execution is done
        print("\n" + "─" * 62)
        print("  [PLANNER]  Reviewing completed execution...")
        print("─" * 62)
        review_prompt = (
            f"You planned a trip to {destination}:\n"
            + "\n".join(f"{i+1}. {s}" for i, s in enumerate(existing_plan))
            + "\n\nExecutor completed:\n"
            + "\n".join(f"- {s[:120]}..." for s in executed)
            + "\n\nGive a short quality review (2-3 sentences)."
        )
        response = llm.invoke([HumanMessage(content=review_prompt)])
        feedback = response.content.strip()
        print(f"  Feedback: {feedback[:200]}")
        return {**state, "feedback": feedback}


# ══════════════════════════════════════════════════════════════════════════════
#  NODE 2 — EXECUTOR  (with real Tavily search)
# ══════════════════════════════════════════════════════════════════════════════
def executor_node(state: TravelPlannerState) -> TravelPlannerState:
    """
    Executes the next plan step.
    If Tavily is available: searches the web for real info, then asks the LLM
                            to synthesize a detailed itinerary from results.
    If Tavily is not available: LLM generates details on its own (v1 behaviour).
    """
    plan         = state["plan"]
    executed     = state.get("executed_steps", [])
    current_step = state.get("current_step", 0)
    destination  = state["destination"]

    if current_step >= len(plan):
        return {**state, "is_complete": True}

    step_title = plan[current_step]
    print("\n" + "─" * 62)
    print(f"  [EXECUTOR]  Step {current_step + 1}/{len(plan)}: '{step_title}'")
    print("─" * 62)

    raw_search_snippets: List[str] = []

    # ── [UPGRADE 1] Real Tavily search ────────────────────────────────────────
    if search_tool:
        query = f"{step_title} travel tips {destination}"
        print(f"  [TAVILY]  Searching: '{query}'")
        try:
            results = search_tool.invoke(query)
            raw_search_snippets = [
                f"[{r.get('title', 'Source')}]: {r.get('content', '')[:300]}"
                for r in results
            ]
            print(f"  [TAVILY]  Got {len(raw_search_snippets)} result(s)")
            for i, snip in enumerate(raw_search_snippets, 1):
                print(f"    {i}. {snip[:100]}...")
        except Exception as e:
            print(f"  [TAVILY]  Search failed ({e}) — falling back to LLM.")

    # ── Build executor prompt ─────────────────────────────────────────────────
    if raw_search_snippets:
        context_block = "\n\n".join(raw_search_snippets)
        execution_prompt = (
            f"You are a travel guide for {destination}.\n"
            f"Plan step: '{step_title}'\n\n"
            f"Real web search results:\n{context_block}\n\n"
            "Using the real info above, write a detailed, actionable itinerary "
            "for this step. Include specific activities, timings, tips, and "
            "highlights from the search results. 4-6 sentences."
        )
    else:
        execution_prompt = (
            f"You are a travel guide for {destination}.\n"
            f"Plan step: '{step_title}'\n\n"
            "Provide a detailed, actionable itinerary for this step. "
            "Include specific activities, timings, tips, and highlights. "
            "4-6 sentences."
        )

    response        = llm.invoke([HumanMessage(content=execution_prompt)])
    execution_detail = response.content.strip()

    print(f"\n  Result preview: {execution_detail[:180]}...")

    new_executed = executed + [
        f"Step {current_step + 1} — {step_title}: {execution_detail}"
    ]

    return {
        **state,
        "executed_steps": new_executed,
        "current_step":   current_step + 1,
        "search_results": raw_search_snippets,
        "is_complete":    len(new_executed) >= len(plan),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CONDITIONAL EDGE — CHECK-PROGRESS
# ══════════════════════════════════════════════════════════════════════════════
def check_progress(state: TravelPlannerState) -> str:
    """
    Routes the graph after each Executor call:
      'next_step' → more steps remain, keep executing
      'review'    → all steps done, send to Planner for final review
      'done'      → Planner reviewed, we are finished
    """
    plan         = state.get("plan", [])
    executed     = state.get("executed_steps", [])
    feedback     = state.get("feedback", "")
    current_step = state.get("current_step", 0)

    print(f"\n  [CHECK-PROGRESS]  {len(executed)}/{len(plan)} steps done", end="")

    if feedback:
        # Planner already reviewed → truly done
        print(" → DONE")
        return "done"

    if len(executed) >= len(plan):
        # All steps executed, needs Planner review
        print(" → send to PLANNER for final review")
        return "review"

    print(f" → EXECUTE step {current_step + 1}")
    return "next_step"


# ══════════════════════════════════════════════════════════════════════════════
#  PRETTY-PRINT RESULTS
# ══════════════════════════════════════════════════════════════════════════════
def print_results(state: TravelPlannerState) -> None:
    bar = "═" * 62
    print(f"\n{bar}")
    print(f"  TRAVEL ITINERARY: {state['destination'].upper()}")
    print(bar)

    print("\n  HIGH-LEVEL PLAN")
    print("  " + "─" * 40)
    for i, s in enumerate(state["plan"], 1):
        print(f"    {i}. {s}")

    print("\n  DETAILED EXECUTION")
    print("  " + "─" * 40)
    for detail in state["executed_steps"]:
        title_part, _, body = detail.partition(": ")
        print(f"\n    > {title_part}")
        # Word-wrap body
        words, line, lines = body.split(), [], []
        for w in words:
            if sum(len(x) + 1 for x in line) + len(w) > 70:
                lines.append(" ".join(line)); line = [w]
            else:
                line.append(w)
        if line: lines.append(" ".join(line))
        for ln in lines:
            print(f"      {ln}")

    feedback = state.get("feedback", "")
    if feedback:
        print("\n  PLANNER FINAL REVIEW")
        print("  " + "─" * 40)
        for ln in feedback.split(". "):
            if ln.strip():
                print(f"    {ln.strip()}.")

    print(f"\n{bar}")
    print("  Itinerary generation complete!")
    print(f"{bar}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  BUILD THE GRAPH
# ══════════════════════════════════════════════════════════════════════════════
def build_graph(memory: MemorySaver, interactive: bool = True) -> StateGraph:
    """
    Builds and compiles the LangGraph workflow.

    Args:
        memory:      MemorySaver checkpointer for state persistence.
        interactive: If True (default), adds interrupt_before=["Executor"]
                     so the graph pauses for Human-in-the-Loop approval.
                     Set False for headless/eval mode (no user prompts).
    """
    graph = StateGraph(TravelPlannerState)

    graph.add_node("Planner",  planner_node)
    graph.add_node("Executor", executor_node)

    graph.set_entry_point("Planner")
    graph.add_edge("Planner", "Executor")

    graph.add_conditional_edges(
        "Executor",
        check_progress,
        {
            "next_step": "Executor",
            "review":    "Planner",
            "done":      END,
        },
    )

    compile_kwargs: dict = {"checkpointer": memory}
    if interactive:
        compile_kwargs["interrupt_after"] = ["Planner"]  # HITL gate: Pause after plan is generated

    return graph.compile(**compile_kwargs)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN RUNNER
# ══════════════════════════════════════════════════════════════════════════════
def run_travel_planner(destination: str, num_days: int = 3, thread_id: Optional[str] = None) -> None:
    """
    Full workflow with all 3 upgrades:
      1. Real Tavily web search in Executor
      2. Human-in-the-Loop approval gate after Planner
      3. MemorySaver persistence — state stored under thread_id
    """
    # [UPGRADE 3] MemorySaver — create once, reuse across resume calls
    memory    = MemorySaver()
    app       = build_graph(memory)
    thread_id = thread_id or str(uuid.uuid4())
    config    = {"configurable": {"thread_id": thread_id}}

    print("\n" + "=" * 62)
    print("  LangGraph Travel Planner Agent")
    print("=" * 62)
    print(f"  Destination : {destination}")
    print(f"  Thread ID   : {thread_id}")
    print(f"  (Re-use this thread_id to resume your session)")
    print("=" * 62 + "\n")

    initial_state: TravelPlannerState = {
        "destination":    destination,
        "num_days":       num_days,
        "plan":           [],
        "executed_steps": [],
        "current_step":   0,
        "feedback":       "",
        "is_complete":    False,
        "human_approved": False,
        "search_results": [],
    }

    # ── Phase 1: Run Planner → graph pauses before Executor ───────────────────
    print("  Phase 1: Generating plan...\n")
    app.invoke(initial_state, config=config)

    # ── [UPGRADE 2] Human-in-the-Loop  ────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  HUMAN-IN-THE-LOOP")
    print("=" * 62)
    print(f"  The Planner has created the {num_days}-step itinerary above.")
    print("  Review it and decide:")
    print()
    while True:
        choice = input("  Type 'approve' to execute OR 'reject' to abort: ").strip().lower()
        if choice in ("approve", "a", "yes", "y"):
            print("\n  Plan APPROVED — starting execution...\n")
            break
        elif choice in ("reject", "r", "no", "n", "abort"):
            print("\n  Plan REJECTED — aborting agent.")
            print(f"  Your session (thread_id={thread_id}) is saved.")
            print("  You can restart with a different destination.\n")
            return
        else:
            print("  Please type 'approve' or 'reject'.")

    # ── Phase 2: Resume graph — Executor runs all steps ────────────────────────
    print("  Phase 2: Executing steps with real web search...\n")
    final_state = app.invoke(None, config=config)   # None = resume from checkpoint

    # ── Print final itinerary ──────────────────────────────────────────────────
    print_results(final_state)

    print(f"  [MEMORY]  Session saved. Resume later with thread_id:")
    print(f"            {thread_id}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  [UPGRADE 5] HEADLESS RUNNER  (used by eval_agent.py)
# ══════════════════════════════════════════════════════════════════════════════
def run_headless(destination: str, num_days: int = 3, thread_id: Optional[str] = None) -> TravelPlannerState:
    """
    Runs the full agent pipeline without any user interaction.
    Human-in-the-Loop is disabled; the plan is auto-approved.
    Used by eval_agent.py for automated batch evaluation.

    Args:
        destination: Travel destination string.
        thread_id:   Optional thread ID for memory persistence.

    Returns:
        The final TravelPlannerState after the graph completes.
    """
    memory    = MemorySaver()
    app       = build_graph(memory, interactive=False)   # no HITL interrupt
    thread_id = thread_id or str(uuid.uuid4())
    config    = {"configurable": {"thread_id": thread_id}}

    initial_state: TravelPlannerState = {
        "destination":    destination,
        "num_days":       num_days,
        "plan":           [],
        "executed_steps": [],
        "current_step":   0,
        "feedback":       "",
        "is_complete":    False,
        "human_approved": True,    # auto-approved in headless mode
        "search_results": [],
    }

    final_state = app.invoke(initial_state, config=config)
    return final_state


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "=" * 62)
    print("  LangGraph Travel Planner")
    print("=" * 62)

    if len(sys.argv) > 1:
        destination = " ".join(sys.argv[1:])
        num_days = 3
    else:
        destination = input("  Enter travel destination: ").strip()
        if not destination:
            destination = "Kyoto, Japan"
            print(f"  (Defaulting to: {destination})")
        
        days_input = input("  Enter number of days (e.g. 5): ").strip()
        num_days = int(days_input) if days_input.isdigit() else 3

    # Optional: reuse a saved session
    print()
    saved_thread = input(
        "  Resume a previous session? Paste thread_id (or press Enter for new): "
    ).strip()
    thread_id = saved_thread if saved_thread else None

    run_travel_planner(destination, num_days=num_days, thread_id=thread_id)
