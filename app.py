import streamlit as st
import uuid
from langgraph.checkpoint.memory import MemorySaver
from planner_agent import build_graph, TravelPlannerState

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Travel Planner", 
    page_icon="🌍", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS for a Colorful GUI ────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: -webkit-linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-size: 3.5rem;
        font-weight: 900;
        margin-bottom: 0px;
    }
    .sub-header {
        color: #A0AEC0;
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .plan-box {
        background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid #4ECDC4;
        margin-bottom: 15px;
        color: #E2E8F0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .exec-box {
        background: linear-gradient(135deg, #2A4365 0%, #1A365D 100%);
        padding: 25px;
        border-radius: 12px;
        border-left: 6px solid #FF6B6B;
        margin-bottom: 15px;
        color: #E2E8F0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        line-height: 1.6;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🌍 AI Travel Planner</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">LangGraph Agent • Groq LLM • Tavily Web Search</div>', unsafe_allow_html=True)

# ── Initialize State ─────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.memory    = MemorySaver()
    st.session_state.graph     = build_graph(st.session_state.memory)
    st.session_state.stage     = "input"  # stages: input -> review -> done

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Graph Memory")
    st.info(f"**Thread ID:**\n`{st.session_state.thread_id}`")
    st.write("This ID persists your LangGraph session state.")
    if st.button("Reset Memory / New Session", type="primary"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.memory    = MemorySaver()
        st.session_state.graph     = build_graph(st.session_state.memory)
        st.session_state.stage     = "input"
        st.rerun()

# ── Stage 1: Input ───────────────────────────────────────────────────────────
if st.session_state.stage == "input":
    st.markdown("### Where to next?")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        dest = st.text_input("Destination", placeholder="e.g., Reykjavik, Iceland")
    with col2:
        days = st.number_input("Days", min_value=1, max_value=14, value=3)
    
    if st.button("Generate High-Level Plan", type="primary", use_container_width=True):
        if dest.strip() == "":
            st.warning("Please enter a destination!")
        else:
            with st.spinner(f"Agent Planner is designing a {days}-day trip to {dest}... 🤖"):
                initial_state = {
                    "destination":    dest,
                    "num_days":       days,
                    "plan":           [],
                    "executed_steps": [],
                    "current_step":   0,
                    "feedback":       "",
                    "is_complete":    False,
                    "human_approved": False,
                    "search_results": []
                }
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                # Hit the Planner node and pause before Executor
                st.session_state.graph.invoke(initial_state, config=config)
                st.session_state.stage = "review"
                st.rerun()

# ── Stage 2: Human-in-the-Loop Review ────────────────────────────────────────
elif st.session_state.stage == "review":
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    state  = st.session_state.graph.get_state(config).values
    
    st.markdown("### 🛑 Human-in-the-Loop Review")
    st.write("The Agent proposes the following itinerary. Approve it to let the Executor scrape the web for real-time details.")
    
    plan = state.get("plan", [])
    for i, step in enumerate(plan, 1):
        st.markdown(f'<div class="plan-box"><strong>Day {i}:</strong> {step}</div>', unsafe_allow_html=True)
        
    st.markdown("#### Do you approve this plan?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Approve & Execute", type="primary", use_container_width=True):
            with st.spinner("Executing... The Agent is searching the web (Tavily) and writing details. This may take a minute! 🌐"):
                
                # Keep invoking until the graph finishes or hits the Final Review planner node.
                # This ensures all days are executed, protecting against intermediate interrupts.
                while True:
                    st.session_state.graph.invoke(None, config=config)
                    current_snap = st.session_state.graph.get_state(config)
                    # If no next nodes are scheduled, or if we hit the final Planner review, break
                    if not current_snap.next or current_snap.next[0] == "Planner":
                        break
                        
                st.session_state.stage = "done"
                st.rerun()
    with col2:
        if st.button("❌ Reject & Edit", use_container_width=True):
            st.session_state.stage = "input"
            st.rerun()

# ── Stage 3: Done ────────────────────────────────────────────────────────────
elif st.session_state.stage == "done":
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    state  = st.session_state.graph.get_state(config).values
    
    st.markdown("### ✨ Final Detailed Itinerary")
    st.success("Execution Complete! The graph has finished and memory is saved.")
    
    executed = state.get("executed_steps", [])
    for i, step in enumerate(executed, 1):
        with st.expander(f"📍 Day {i} Execution Details", expanded=True):
            st.markdown(f'<div class="exec-box">{step}</div>', unsafe_allow_html=True)
            
    st.write("---")
    if st.button("Plan Another Trip", type="primary"):
        st.session_state.stage = "input"
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()
