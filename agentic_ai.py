"""
SENTINEL Agentic AI — Multi-step ReAct reasoning with streaming.
Uses shared models. Gathers data in phases, reasons iteratively, produces report.
"""

import os
import json
from dotenv import load_dotenv
from models import get_llm, get_total_complaints

from queries import (
    demographic_breakdown,
    geographic_breakdown,
    complaint_type_analysis,
    temporal_decay_analysis,
    get_silenced_complaints
)

load_dotenv()

print("=" * 60)
print(" INITIALIZING AGENTIC AI SYSTEM")
print("=" * 60)
print("* AI system ready (ReAct reasoning engine)")

# ─── Tool registry for the agent ───────────────────────────────

AGENT_TOOLS = {
    "demographics": {
        "fn": lambda: demographic_breakdown(),
        "desc": "Get silence breakdown by gender, caste, income",
    },
    "geography": {
        "fn": lambda: geographic_breakdown(top_n=10),
        "desc": "Get top-10 silenced wards and geographic patterns",
    },
    "categories": {
        "fn": lambda: complaint_type_analysis(),
        "desc": "Get silence by complaint category",
    },
    "temporal": {
        "fn": lambda: temporal_decay_analysis(),
        "desc": "Get temporal decay of complaint handling",
    },
    "critical": {
        "fn": lambda: get_silenced_complaints(threshold=80, limit=10),
        "desc": "Get most critically silenced complaints",
    },
}


# ─── ReAct step: single reasoning iteration ────────────────────

def _react_step(llm, gathered_data: dict, step_num: int, previous_thoughts: list) -> dict:
    """One ReAct iteration: THOUGHT → ACTION → OBSERVATION."""
    tools_available = [t for t in AGENT_TOOLS if t not in gathered_data]

    prompt = f"""You are a forensic data analyst investigating institutional bias.

STEP {step_num} of a multi-step investigation.

DATA GATHERED SO FAR:
{json.dumps(list(gathered_data.keys()))}

TOOLS STILL AVAILABLE:
{json.dumps({t: AGENT_TOOLS[t]['desc'] for t in tools_available})}

PREVIOUS THOUGHTS:
{chr(10).join(previous_thoughts) if previous_thoughts else 'None yet.'}

INSTRUCTIONS:
1. Think about what data you still need.
2. Pick ONE tool to call next, OR say "DONE" if you have enough data.
3. Respond in this EXACT JSON format:
{{"thought": "your reasoning", "action": "tool_name_or_DONE"}}
"""
    response = llm.invoke(prompt)
    text = response.content.strip()

    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        parsed = json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        parsed = {"thought": text, "action": "DONE"}

    return parsed


# ─── Main report generator with ReAct loop ─────────────────────

def generate_bias_report(stream_callback=None) -> str:
    """
    Generate comprehensive bias report using multi-step ReAct reasoning.
    
    Args:
        stream_callback: Optional callable(step_type, message) for streaming updates.
    
    Returns:
        Complete analysis report text.
    """
    def _emit(step_type, msg):
        print(f"  [{step_type}] {msg}")
        if stream_callback:
            stream_callback(step_type, msg)

    _emit("START", "Autonomous bias investigation initiated")

    llm = get_llm()
    gathered_data = {}
    thoughts = []
    max_steps = len(AGENT_TOOLS) + 1

    # ── Phase 1: ReAct data-gathering loop ──
    for step in range(1, max_steps + 1):
        _emit("THINK", f"Step {step}: reasoning about next action...")

        result = _react_step(llm, gathered_data, step, thoughts)
        thought = result.get("thought", "")
        action = result.get("action", "DONE")

        thoughts.append(f"Step {step}: {thought}")
        _emit("THOUGHT", thought)

        if action == "DONE" or action not in AGENT_TOOLS:
            _emit("DECIDE", "Agent determined it has enough data. Synthesizing report.")
            break

        _emit("ACTION", f"Calling tool: {action}")
        try:
            data = AGENT_TOOLS[action]["fn"]()
            gathered_data[action] = data
            _emit("OBSERVATION", f"Got {action} data ({len(json.dumps(data, default=str))} chars)")
        except Exception as e:
            gathered_data[action] = {"error": str(e)}
            _emit("ERROR", f"Tool {action} failed: {e}")

    # Fill in anything the agent skipped
    for tool_name in AGENT_TOOLS:
        if tool_name not in gathered_data:
            try:
                gathered_data[tool_name] = AGENT_TOOLS[tool_name]["fn"]()
            except Exception:
                pass

    # ── Phase 2: Synthesize the report ──
    _emit("SYNTHESIZE", "Generating comprehensive analysis report...")

    total = get_total_complaints()
    demo = gathered_data.get("demographics", {})
    geo = gathered_data.get("geography", {})
    cats = gathered_data.get("categories", [])
    temporal = gathered_data.get("temporal", [])
    critical = gathered_data.get("critical", [])

    top_wards = geo.get("top_silenced", [])[:5] if isinstance(geo, dict) else []
    bottom_wards = (geo.get("all_wards", [])[-3:] if isinstance(geo, dict) else [])

    synthesis_prompt = f"""SILENCE INDEX — FORENSIC BIAS INVESTIGATION

Total complaints analysed: {total}

DEMOGRAPHIC DATA:
{json.dumps(demo, indent=2, default=str)}

GEOGRAPHIC DATA (top 5 worst wards):
{json.dumps(top_wards, indent=2, default=str)}

GEOGRAPHIC DATA (3 best wards):
{json.dumps(bottom_wards, indent=2, default=str)}

COMPLAINT CATEGORIES:
{json.dumps(cats[:5] if isinstance(cats, list) else cats, indent=2, default=str)}

TEMPORAL DECAY:
{json.dumps(temporal, indent=2, default=str)}

CRITICAL CASES (top 5):
{json.dumps([{{'text': c.get('text','')[:120], 'silence_score': c.get('silence_score',0), 'days': c.get('days_in_system',0), 'status': c.get('response_status','')}} for c in (critical[:5] if isinstance(critical, list) else [])], indent=2, default=str)}

AGENT REASONING TRAIL:
{chr(10).join(thoughts)}

Write a comprehensive, markdown-formatted report including:
1. **Executive Summary** — 3-sentence verdict on institutional bias
2. **Demographic Disparities** — gender, caste, income with disparity ratios (e.g. "2.1x")
3. **Geographic Hotspots** — worst wards vs best, zone analysis
4. **Category Analysis** — which complaint types are most ignored
5. **Temporal Decay** — how complaints rot over time
6. **Critical Cases** — 3 most alarming specific complaints
7. **Recommendations** — 5 actionable systemic reforms

Be specific with numbers. Use markdown headers, bullets, and **bold** for emphasis.
"""

    response = llm.invoke(synthesis_prompt)
    report = response.content

    _emit("COMPLETE", "Report generated successfully")
    return report


# MAIN

if __name__ == "__main__":
    try:
        report = generate_bias_report()

        print("\n" + "=" * 60)
        print(" AI ANALYSIS REPORT")
        print("=" * 60)
        print(report)

        with open("AI_BIAS_REPORT.md", "w", encoding="utf-8") as f:
            f.write(report)
        print("\n* Report saved to AI_BIAS_REPORT.md")

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
