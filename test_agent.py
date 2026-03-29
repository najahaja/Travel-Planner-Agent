"""Quick smoke-test: verify all nodes fire and graph completes."""
import sys
sys.path.insert(0, ".")
from planner_agent import run_travel_planner

state = run_travel_planner("Bali, Indonesia")
assert len(state["plan"]) == 3, "Plan must have exactly 3 steps"
assert len(state["executed_steps"]) == 3, "All 3 steps must be executed"
assert state["is_complete"] is True, "is_complete must be True"
print("\n✅ All assertions passed!")
print(f"   Plan steps: {len(state['plan'])}")
print(f"   Executed:   {len(state['executed_steps'])}")
print(f"   Complete:   {state['is_complete']}")
