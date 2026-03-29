"""
=============================================================
  Travel Planner Agent — Evaluation Script
=============================================================
  Runs the agent against 10 diverse destinations and measures:
    • Success Rate      — did it complete without errors?
    • Plan Completeness — did it produce exactly 3 steps?
    • Exec Completeness — were all 3 steps executed?
    • Relevance Score   — do outputs mention the destination?
                          (proxy hallucination check)
    • Step Substance    — are responses detailed (>150 chars)?
    • Latency           — seconds per destination
    • Total Cost Proxy  — estimated LLM calls made

  Output: a formatted report table + aggregate summary saved
          to eval_results.txt

  Run:
    python -X utf8 eval_agent.py
=============================================================
"""

import os
import sys
import time
import json
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

# ── Force UTF-8 on Windows ────────────────────────────────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Suppress noisy INFO logs from the agent during eval ───────────────────────
os.environ["PYTHONWARNINGS"] = "ignore"

# ── Import the headless runner from the v2 agent ──────────────────────────────
# This must come AFTER the env suppression so we don't see Tavily/LangSmith
# status messages for every single run.
from planner_agent import run_headless, TravelPlannerState


# ══════════════════════════════════════════════════════════════════════════════
#  EVAL CONFIG — 10 DIVERSE TEST DESTINATIONS
# ══════════════════════════════════════════════════════════════════════════════
TEST_DESTINATIONS = [
    "Paris, France",
    "Tokyo, Japan",
    "New York City, USA",
    "Colombo, Sri Lanka",
    "Dubai, UAE",
    "Rome, Italy",
    "Cape Town, South Africa",
    "Sydney, Australia",
    "Istanbul, Turkey",
    "Bali, Indonesia",
]

# Minimum character length for a step to be considered "substantial"
MIN_STEP_LENGTH = 150

# Minimum expected LLM calls per run (1 plan + 3 execute + 1 review)
EXPECTED_LLM_CALLS = 5

# Rate-limit buffer between runs (seconds) — avoids Groq 429 errors
RATE_LIMIT_PAUSE = 2.0


# ══════════════════════════════════════════════════════════════════════════════
#  RESULT DATA CLASS
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class EvalResult:
    destination:      str
    success:          bool
    latency_s:        float
    plan_steps:       int   = 0
    executed_steps:   int   = 0
    plan_complete:    bool  = False    # plan_steps == 3
    exec_complete:    bool  = False    # executed_steps == 3
    relevance_score:  float = 0.0     # fraction of steps mentioning destination
    avg_step_len:     float = 0.0     # avg chars per executed step
    all_substantial:  bool  = False    # all steps > MIN_STEP_LENGTH chars
    error:            str   = ""


# ══════════════════════════════════════════════════════════════════════════════
#  METRICS HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _relevance_keywords(destination: str) -> List[str]:
    """Extract search keywords from a destination string."""
    # Split "City, Country" into individual words (lowercased), drop tiny words
    parts = destination.replace(",", " ").split()
    return [p.lower() for p in parts if len(p) > 2]


def _relevance_score(executed_steps: List[str], destination: str) -> float:
    """
    For each executed step, check whether at least one keyword from the
    destination name appears in the step text.
    Returns fraction of steps that pass (0.0 – 1.0).
    Serves as a simple anti-hallucination proxy.
    """
    if not executed_steps:
        return 0.0
    keywords = _relevance_keywords(destination)
    hits = 0
    for step in executed_steps:
        step_lower = step.lower()
        if any(kw in step_lower for kw in keywords):
            hits += 1
    return round(hits / len(executed_steps), 2)


def _avg_step_length(executed_steps: List[str]) -> float:
    if not executed_steps:
        return 0.0
    return round(sum(len(s) for s in executed_steps) / len(executed_steps), 1)


# ══════════════════════════════════════════════════════════════════════════════
#  SINGLE-DESTINATION EVAL
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_one(destination: str, index: int, total: int) -> EvalResult:
    """Run the agent on one destination and return an EvalResult."""
    prefix = f"  [{index:02d}/{total}]  {destination:<30}"
    print(f"{prefix} running...", end="", flush=True)

    t0 = time.perf_counter()
    try:
        state      = run_headless(destination)
        latency_s  = round(time.perf_counter() - t0, 2)

        plan       = state.get("plan", [])
        executed   = state.get("executed_steps", [])
        rel_score  = _relevance_score(executed, destination)
        avg_len    = _avg_step_length(executed)
        substantial = all(len(s) >= MIN_STEP_LENGTH for s in executed)

        result = EvalResult(
            destination     = destination,
            success         = True,
            latency_s       = latency_s,
            plan_steps      = len(plan),
            executed_steps  = len(executed),
            plan_complete   = len(plan) == 3,
            exec_complete   = len(executed) == 3,
            relevance_score = rel_score,
            avg_step_len    = avg_len,
            all_substantial = substantial,
        )
        status = "PASS" if (result.plan_complete and result.exec_complete) else "WARN"
        print(f" {status}  ({latency_s}s)")
        return result

    except Exception as exc:
        latency_s = round(time.perf_counter() - t0, 2)
        err_msg   = str(exc)[:120]
        print(f" FAIL  ({latency_s}s)  Error: {err_msg}")
        return EvalResult(
            destination = destination,
            success     = False,
            latency_s   = latency_s,
            error       = err_msg,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  REPORT PRINTER
# ══════════════════════════════════════════════════════════════════════════════
def _bar(n: int = 78) -> str:
    return "═" * n

def _sep(n: int = 78) -> str:
    return "─" * n

def print_report(results: List[EvalResult], model: str = "llama-3.3-70b-versatile") -> str:
    """Prints the full evaluation report and returns it as a string."""
    lines = []
    def p(s: str = "") -> None:
        lines.append(s)
        print(s)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = len(results)
    successes = [r for r in results if r.success]
    passes    = [r for r in results if r.plan_complete and r.exec_complete]

    p()
    p(_bar())
    p("  TRAVEL PLANNER AGENT  —  EVALUATION REPORT")
    p(f"  Run date : {now}")
    p(f"  Model    : {model}")
    p(f"  Tested   : {total} destinations")
    p(_bar())

    # ── Per-destination table ─────────────────────────────────────────────────
    p()
    p("  PER-DESTINATION RESULTS")
    p("  " + _sep(76))
    header = (
        f"  {'Destination':<28} {'OK?':<5} {'Plan':<5} {'Exec':<5} "
        f"{'Relev':>6} {'Substance':>10} {'Latency':>8}"
    )
    p(header)
    p("  " + _sep(76))

    for r in results:
        ok_str  = "PASS" if (r.plan_complete and r.exec_complete) else ("FAIL" if not r.success else "WARN")
        plan_s  = f"{r.plan_steps}/3"    if r.success else "—"
        exec_s  = f"{r.executed_steps}/3" if r.success else "—"
        rel_s   = f"{int(r.relevance_score*100)}%"  if r.success else "—"
        subst_s = "YES" if r.all_substantial else ("NO" if r.success else "—")
        lat_s   = f"{r.latency_s}s"

        row = (
            f"  {r.destination:<28} {ok_str:<5} {plan_s:<5} {exec_s:<5} "
            f"{rel_s:>6} {subst_s:>10} {lat_s:>8}"
        )
        p(row)

        if r.error:
            p(f"    ERROR: {r.error}")

    p("  " + _sep(76))

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    success_rate  = round(len(successes) / total * 100, 1)
    pass_rate     = round(len(passes) / total * 100, 1)
    plan_acc      = round(sum(r.plan_complete for r in results) / total * 100, 1)
    exec_acc      = round(sum(r.exec_complete for r in results) / total * 100, 1)
    avg_rel       = round(sum(r.relevance_score for r in successes) / max(len(successes), 1) * 100, 1)
    avg_lat       = round(sum(r.latency_s for r in results) / total, 2)
    total_dur     = round(sum(r.latency_s for r in results), 1)
    avg_len       = round(sum(r.avg_step_len for r in successes) / max(len(successes), 1), 0)
    subst_rate    = round(sum(r.all_substantial for r in successes) / max(len(successes), 1) * 100, 1)

    p()
    p("  AGGREGATE METRICS")
    p("  " + _sep(44))
    metrics = [
        ("Destinations tested",       f"{total}"),
        ("Success rate",              f"{success_rate}%   ({len(successes)}/{total} completed)"),
        ("Full pass rate",            f"{pass_rate}%   ({len(passes)}/{total} perfect)"),
        ("Plan accuracy (3 steps)",   f"{plan_acc}%"),
        ("Execution accuracy (3/3)",  f"{exec_acc}%"),
        ("Avg relevance / anti-halluc",f"{avg_rel}%"),
        ("Substance rate (>150 chars)",f"{subst_rate}%"),
        ("Avg step length (chars)",   f"{int(avg_len)}"),
        ("Avg latency per dest",      f"{avg_lat}s"),
        ("Total evaluation time",     f"{total_dur}s"),
    ]
    for label, value in metrics:
        p(f"    {label:<32} {value}")

    p()

    # ── Grade ─────────────────────────────────────────────────────────────────
    score = (success_rate + plan_acc + exec_acc + avg_rel + subst_rate) / 5
    if   score >= 90: grade = "A  (Excellent)"
    elif score >= 75: grade = "B  (Good)"
    elif score >= 60: grade = "C  (Acceptable)"
    else:             grade = "D  (Needs improvement)"

    p("  " + _sep(44))
    p(f"    Overall Score : {score:.1f}/100   Grade: {grade}")
    p("  " + _sep(44))
    p()
    p(_bar())
    p("  Evaluation complete.")
    p(_bar())
    p()

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def run_evaluation(
    destinations: List[str] = TEST_DESTINATIONS,
    save_path: str = "eval_results.txt",
    save_json: str = "eval_results.json",
) -> List[EvalResult]:
    """
    Main eval loop.
    Args:
        destinations: List of destinations to test.
        save_path:    File to write the human-readable report.
        save_json:    File to write machine-readable JSON results.
    Returns:
        List of EvalResult objects.
    """
    total = len(destinations)
    print()
    print("=" * 78)
    print(f"  EVAL STARTED  |  {total} destinations  |  {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 78)
    print(f"  (Pausing {RATE_LIMIT_PAUSE}s between runs to avoid API rate limits)")
    print()

    results: List[EvalResult] = []
    for i, dest in enumerate(destinations, 1):
        result = evaluate_one(dest, i, total)
        results.append(result)
        if i < total:
            time.sleep(RATE_LIMIT_PAUSE)

    # ── Print + save report ───────────────────────────────────────────────────
    report_str = print_report(results)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report_str)
    print(f"  Report saved to: {save_path}")

    # ── Save machine-readable JSON ────────────────────────────────────────────
    json_data = {
        "timestamp": datetime.now().isoformat(),
        "model":     "llama-3.3-70b-versatile",
        "total":     total,
        "results": [
            {
                "destination":      r.destination,
                "success":          r.success,
                "latency_s":        r.latency_s,
                "plan_steps":       r.plan_steps,
                "executed_steps":   r.executed_steps,
                "plan_complete":    r.plan_complete,
                "exec_complete":    r.exec_complete,
                "relevance_score":  r.relevance_score,
                "avg_step_len":     r.avg_step_len,
                "all_substantial":  r.all_substantial,
                "error":            r.error,
            }
            for r in results
        ],
    }
    with open(save_json, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    print(f"  JSON data saved to: {save_json}")
    print()

    return results


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Optional: test only a subset via CLI argument
    # e.g.  python eval_agent.py 3   → only first 3 destinations
    subset = int(sys.argv[1]) if len(sys.argv) > 1 else len(TEST_DESTINATIONS)
    run_evaluation(TEST_DESTINATIONS[:subset])
