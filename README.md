# 🌍 AI Travel Planner Agent

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![LangGraph](https://img.shields.io/badge/LangGraph-00C4B6?style=for-the-badge&logo=langchain&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-F55036?style=for-the-badge)
[![Live Demo](https://img.shields.io/badge/🟢_Play_Live_Demo-Streamlit_Cloud-success?style=for-the-badge)](https://travel-planner-agent-1.streamlit.app/)

A production-grade, multi-node **LangGraph** agent that takes a travel destination and produces a fully detailed, dynamic day-by-day itinerary. Deployed via Streamlit Community Cloud.

---

## Architecture

```
START
  │
  ▼
┌──────────┐
│  Planner │  → generates the high-level 3-step plan
└──────────┘
  │
  ▼
[PAUSE — Human-in-the-Loop]
  │
  │  user types "approve"
  ▼
┌──────────┐   Tavily search   ┌────────────────┐
│ Executor │ ───────────────▶  │  Web Search    │
└──────────┘                   └────────────────┘
  │
  ▼
[Check-Progress]
  ├── next_step → Executor (loop)
  ├── review    → Planner (final review)
  └── done      → END
```

## Features

1. **Interactive Streamlit GUI**: A sleek, colorful Web UI built with Streamlit that hooks directly into the LangGraph state.
2. **Dynamic Itinerary Length**: Instead of a hardcoded plan, you can specify exactly how many days you want the trip to be. The agent adapts its state and execution loops dynamically.
3. **Real Tool Integration (Tavily Web Search)**: Computes dynamic information safely by retrieving real web context directly on each plan step.
4. **Human-in-the-Loop**: Execution pauses to show the proposed plan to the user via UI components. Click `approve` to execute or `reject` to abort.
5. **Memory Persistence**: The session is stored in memory via a `thread_id`. You can pick right back up where you left off.
6. **LangSmith Tracing**: Full visual tracing out-of-the-box (if `LANGCHAIN_API_KEY` is present in your `.env`).
7. **Evaluation Suite**: Ships with `eval_agent.py` to test performance, hallucination proxy, and latency across 10 diverse destinations.

---

## Getting Started

### 1. Setup & Installation

Clone the repository and install the dependencies from `requirements.txt`:

```powershell
# 1. Clone the repo
git clone https://github.com/najahaja/Travel-Planner-Agent.git
cd Travel-Planner-Agent

# 2. Create and activate a Virtual Environment (Recommended on Windows)
python -m venv venv
.\venv\Scripts\activate

# 3. Install all dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

Create an `.env` file in the root directory:

```env
GROQ_API_KEY=gsk_...
TAVILY_API_KEY=tvly-...
LANGCHAIN_API_KEY=lsv2_...
LANGCHAIN_PROJECT=travel-planner-agent
```

- **Groq**: Required. LLM engine.
- **Tavily**: Optional but recommended. Needed for actual web search, otherwise the execution runs purely via LLM knowledge.
- **LangSmith**: Optional. Traces will quietly disable if absent.

### 3. Running the Agent

You can start the agent interactivly:

```powershell
python -X utf8 planner_agent.py 
```

Or pass a destination directly:

```powershell
python -X utf8 planner_agent.py "Colombo, Sri Lanka"
```

> **Note:** The `-X utf8` flag prevents terminal encoding errors on Windows systems with emoji characters.

### 4. Running the Streamlit Web App

To launch the interactive GUI:

```powershell
streamlit run app.py
```

### 5. Running the Evaluation Script

Run full headless evaluation against 10 destinations to get latency and scoring:

```powershell
python -X utf8 eval_agent.py
```

---

## Challenges & Debugging

As this project scaled from a single prompt to a cyclic multi-node graph, a major challenge was **infinite looping**.

### The Problem: Infinite Graph Execution
Early in development, the `Executor` node would sometimes get stuck in an infinite loop, endlessly generating activities without ever reaching the `done` state. Because LangGraph defaults to continuing execution until explicitly hitting the `END` node, a bug in the state mutation (specifically, the `current_step` counter failing to increment during a failed Tavily search) caused the conditional edge tracker to repeatedly return `"execute"`.

### The Solution: A Safer Conditional Edge Constraint
To fix this, I redesigned the `Check-Progress` edge. Instead of blindly trusting a `current_step` counter, I introduced a deterministic constraints pattern:

```python
def check_progress(state: TravelPlannerState) -> str:
    plan = state.get("plan", [])
    executed = state.get("executed_steps", [])
    
    # 1. Fallback / Max Iterations Constraint
    if len(executed) >= len(plan):
        return "review" # Force break out of Executor loop
        
    return "next_step"
```

By safely checking the *length* of the completed tasks directly against the *target length* of the plan, the graph is structurally guaranteed to break out of the loop and route to the final `Planner` review. This ensures safety without having to set an arbitrary maximum recursion limit on the entire application.

### The Problem: Streamlit Statelessness vs LangGraph Interrupts
When migrating the terminal script to a **Streamlit GUI**, a new issue arose. LangGraph's `interrupt_after` pauses the graph and returns control to the user. But because Streamlit is stateless and refreshes top-to-bottom on every interaction, invoking the graph after a pause would only run *one* step of the iteration before giving control back to Streamlit, effectively stalling the UI mid-execution.

### The Solution: Auto-Resume Invocation Loop
To fix this, I wrapped the resumption call inside the Streamlit cache in a `while True` loop that interrogates the graph state. It repeatedly invokes the graph behind a UI loading spinner until it confirms the graph has finished scheduling `Executor` nodes and is ready for the final review.

```python
# Keep invoking until the graph finishes or hits the Final Review node.
# This ensures all days are executed, protecting against intermediate interrupts.
while True:
    st.session_state.graph.invoke(None, config=config)
    current_snap = st.session_state.graph.get_state(config)
    if not current_snap.next or current_snap.next[0] == "Planner":
        break
```
This pattern allows LangGraph to maintain strict backend checkpointing while giving the Streamlit frontend a smooth, unbroken loading experience.

---

## 🌟 Support & Contributions

If you find this project helpful or plan to use it as a template for your own Agentic workflows, **please consider leaving a star (⭐️) on this repository!** It helps others discover this project.

Contributions, issues, and feature requests are welcome. Feel free to check the issues page if you want to contribute.
