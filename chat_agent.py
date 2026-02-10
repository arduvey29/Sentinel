"""
SENTINEL Chat Agent — conversation memory + Gemini function-calling.
Uses shared models from models.py.
"""

import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from models import qdrant, get_llm, get_embedding_model, get_total_complaints, CHAT_COLLECTION

load_dotenv()

# QDRANT CHAT HISTORY MANAGEMENT

def init_chat_collection():
    """Initialize Qdrant collection for chat history"""
    collections = [c.name for c in qdrant.get_collections().collections]
    
    if CHAT_COLLECTION not in collections:
        qdrant.create_collection(
            collection_name=CHAT_COLLECTION,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        print(f"[CHAT] Created collection: {CHAT_COLLECTION}")
    return True

def create_session(name: Optional[str] = None) -> str:
    """Create a new chat session"""
    session_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().isoformat()
    
    if not name:
        name = f"Investigation {timestamp[:10]}"
    
    # Store session metadata as first message
    store_message(
        session_id=session_id,
        role="system",
        content=f"Session created: {name}",
        metadata={"session_name": name, "created_at": timestamp}
    )
    
    return session_id

def store_message(
    session_id: str,
    role: str,
    content: str,
    metadata: Optional[Dict] = None
) -> str:
    """Store a chat message in Qdrant with embedding"""
    init_chat_collection()
    
    message_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    
    # Generate embedding
    model = get_embedding_model()
    embedding = model.encode(content).tolist()
    
    # Build payload
    payload = {
        "message_id": message_id,
        "session_id": session_id,
        "role": role,
        "content": content,
        "timestamp": timestamp,
        **(metadata or {})
    }
    
    # Store in Qdrant
    qdrant.upsert(
        collection_name=CHAT_COLLECTION,
        points=[PointStruct(
            id=message_id,
            vector=embedding,
            payload=payload
        )]
    )
    
    return message_id

def get_session_history(session_id: str, limit: int = 50) -> List[Dict]:
    """Retrieve chat history for a session"""
    init_chat_collection()
    
    results = qdrant.scroll(
        collection_name=CHAT_COLLECTION,
        scroll_filter=Filter(
            must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))]
        ),
        limit=limit,
        with_payload=True,
        with_vectors=False
    )
    
    messages = [point.payload for point in results[0]]
    # Sort by timestamp
    messages.sort(key=lambda x: x.get("timestamp", ""))
    
    return messages

def search_chat_history(query: str, session_id: Optional[str] = None, limit: int = 10) -> List[Dict]:
    """Semantic search across chat history"""
    init_chat_collection()
    
    model = get_embedding_model()
    query_vector = model.encode(query).tolist()
    
    search_filter = None
    if session_id:
        search_filter = Filter(
            must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))]
        )
    
    results = qdrant.query_points(
        collection_name=CHAT_COLLECTION,
        query=query_vector,
        query_filter=search_filter,
        limit=limit,
        with_payload=True
    )
    
    return [{"score": r.score, **r.payload} for r in results.points]

def list_sessions(limit: int = 20) -> List[Dict]:
    """List all chat sessions"""
    init_chat_collection()
    
    results = qdrant.scroll(
        collection_name=CHAT_COLLECTION,
        scroll_filter=Filter(
            must=[FieldCondition(key="role", match=MatchValue(value="system"))]
        ),
        limit=limit,
        with_payload=True,
        with_vectors=False
    )
    
    sessions = []
    for point in results[0]:
        payload = point.payload
        sessions.append({
            "session_id": payload.get("session_id"),
            "name": payload.get("session_name", "Unnamed"),
            "created_at": payload.get("created_at", payload.get("timestamp"))
        })
    
    sessions.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return sessions

# DATA QUERY TOOLS

from queries import (
    demographic_breakdown,
    geographic_breakdown,
    complaint_type_analysis,
    temporal_decay_analysis,
    similarity_search,
    get_silenced_complaints,
    cross_tabulation,
    field_breakdown,
    filtered_breakdown,
)

def tool_get_demographics(category: str = "all") -> Dict:
    """Get silence breakdown by demographics (gender/caste/income)"""
    data = demographic_breakdown()
    if category == "gender":
        return {"by_gender": data["by_gender"]}
    elif category == "caste":
        return {"by_caste": data["by_caste"]}
    elif category == "income":
        return {"by_income": data["by_income"]}
    return data

def tool_get_geography(top_n: int = 10) -> Dict:
    """Get ward-level silence analysis"""
    return geographic_breakdown()

def tool_get_categories() -> List[Dict]:
    """Get silence breakdown by complaint category"""
    return complaint_type_analysis()

def tool_get_temporal() -> Dict:
    """Get temporal decay analysis"""
    return temporal_decay_analysis()

def tool_get_stats() -> Dict:
    """Get overall statistics"""
    silenced = get_silenced_complaints(threshold=70, limit=10000)
    total_silenced = len(silenced)
    total = get_total_complaints()
    
    avg_silence = sum(c['silence_score'] for c in silenced) / len(silenced) if silenced else 0
    avg_days = sum(c['days_in_system'] for c in silenced) / len(silenced) if silenced else 0
    
    return {
        "total_complaints": total,
        "total_silenced": total_silenced,
        "silence_rate": round((total_silenced / total) * 100, 1) if total else 0,
        "avg_silence_score": round(avg_silence, 1),
        "avg_days_in_system": round(avg_days, 0)
    }

def tool_get_critical(threshold: int = 80, limit: int = 10) -> Dict:
    """Get critically silenced complaints"""
    complaints = get_silenced_complaints(threshold=threshold, limit=limit)
    return {
        "total": len(complaints),
        "threshold": threshold,
        "complaints": complaints[:limit]
    }

def tool_search_complaints(query: str = "", limit: int = 10) -> Dict:
    """Semantic search for complaints"""
    if not query:
        return {"error": "No search query provided", "results": []}
    results = similarity_search(query, limit=limit)
    return {"query": query, "total": len(results), "results": results}


def tool_filtered_query(message: str) -> Dict:
    """
    LLM-powered compound query: parse user message into filters + breakdown,
    then run filtered_breakdown() with the parsed spec.
    Handles queries like 'water complaints in Ward 5 vs Ward 45 by caste'.
    """
    parse_prompt = f"""You parse natural language queries about civic complaint data into structured filters.

AVAILABLE PAYLOAD FIELDS:
- category: Water Supply, Roads, Sanitation, Electricity, Waste Management, Public Transport, Safety, Health
- ward: Ward 1 through Ward 50
- ward_type: Elite, Middle, Poor, Slum
- gender: M, F, Other
- caste: General, OBC, SC, ST
- income_bracket: 0-3L, 3-6L, 6-10L, 10L+
- response_status: NO_RESPONSE, RESPONDED, RESOLVED, REJECTED

User query: "{message}"

Respond ONLY with a JSON object (no markdown):
{{
  "filters": {{"field": "value" or "field": ["val1", "val2"]}},
  "breakdown_field": "<field to group results by>",
  "cross_field": "<optional second grouping field or null>",
  "metric": "avg_silence or count or silenced_pct"
}}

Rules:
- "water complaints" → filter: {{"category": "Water Supply"}}
- "in Ward 5 vs Ward 45" → filter: {{"ward": ["Ward 5", "Ward 45"]}}
- "by caste" or "caste pattern" → breakdown_field: "caste"
- If comparing two wards, use breakdown_field for the ward and cross_field for the demographic.
- If only filtering + one breakdown, set cross_field to null.
- Default metric: "avg_silence".
- Match category names exactly (Water Supply, not Water)."""

    spec = None
    try:
        llm = get_llm()
        from langchain_core.messages import HumanMessage
        resp = llm.invoke([HumanMessage(content=parse_prompt)])
        text = resp.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        spec = json.loads(text.strip())
        print(f"[FILTERED QUERY] LLM spec: {spec}")
    except Exception as e:
        print(f"[FILTERED QUERY] LLM failed ({e}), using keyword fallback")
        spec = _filtered_query_fallback(message)

    if not spec:
        return {"error": "Could not parse query into filters"}

    try:
        result = filtered_breakdown(
            breakdown_field=spec.get("breakdown_field", "caste"),
            filters=spec.get("filters"),
            cross_field=spec.get("cross_field"),
            metric=spec.get("metric", "avg_silence"),
        )
        result["parsed_spec"] = spec
        return result
    except Exception as e:
        return {"error": f"Filtered query failed: {str(e)}", "parsed_spec": spec}


def _filtered_query_fallback(message: str) -> Optional[Dict]:
    """Keyword fallback for compound query parsing."""
    msg = message.lower()
    filters = {}
    breakdown_field = None
    cross_field = None

    # Detect category filters
    cat_map = {
        "water": "Water Supply", "road": "Roads", "roads": "Roads",
        "sanitation": "Sanitation", "electricity": "Electricity",
        "waste": "Waste Management", "transport": "Public Transport",
        "safety": "Safety", "health": "Health",
    }
    for kw, cat in cat_map.items():
        if kw in msg:
            filters["category"] = cat
            break

    # Detect ward filters (Ward N patterns)
    import re
    wards = re.findall(r'ward\s*(\d+)', msg)
    if wards:
        ward_list = [f"Ward {w}" for w in wards]
        filters["ward"] = ward_list if len(ward_list) > 1 else ward_list[0]

    # Detect ward_type
    for zt in ['elite', 'middle', 'poor', 'slum']:
        if zt in msg:
            filters["ward_type"] = zt.capitalize()
            break

    # Detect status
    for st in ['no_response', 'responded', 'resolved', 'rejected']:
        if st.replace('_', ' ') in msg or st in msg:
            filters["response_status"] = st.upper()
            break

    # Detect breakdown/cross fields
    field_kws = [
        ("caste", ["caste", "sc", "st", "obc"]),
        ("gender", ["gender", "male", "female"]),
        ("income_bracket", ["income", "poor", "rich"]),
        ("ward", ["ward"]),
        ("category", ["category", "type"]),
        ("ward_type", ["zone"]),
        ("response_status", ["status"]),
    ]
    detected = []
    for field, kws in field_kws:
        # only count as breakdown if mentioned in breakdown context
        pattern_words = [' by ' + k for k in kws] + [k + ' pattern' for k in kws] + [k + ' breakdown' for k in kws]
        if any(p in msg for p in pattern_words):
            detected.append(field)

    if detected:
        breakdown_field = detected[0]
        if len(detected) > 1:
            cross_field = detected[1]

    # If wards are being compared, use ward as breakdown, detected field as cross
    if wards and len(wards) > 1:
        if breakdown_field and breakdown_field != "ward":
            cross_field = breakdown_field
        breakdown_field = "ward"

    if not breakdown_field:
        breakdown_field = "caste"  # sensible default

    return {
        "filters": filters if filters else None,
        "breakdown_field": breakdown_field,
        "cross_field": cross_field,
        "metric": "avg_silence",
    }

# ─── PALETTE ────────────────────────────────────────────────
CHART_COLORS = [
    "#00d4ff", "#ff6b6b", "#ffb700", "#00ff88",
    "#a855f7", "#f97316", "#ec4899", "#14b8a6",
    "#6366f1", "#eab308", "#ef4444", "#22d3ee",
]

FIELD_ALIASES = {
    "gender": "gender", "male": "gender", "female": "gender", "women": "gender", "men": "gender",
    "caste": "caste", "sc": "caste", "st": "caste", "obc": "caste", "general": "caste",
    "income": "income_bracket", "income_bracket": "income_bracket", "poor": "income_bracket",
    "rich": "income_bracket", "money": "income_bracket", "wealth": "income_bracket",
    "ward": "ward", "area": "ward", "location": "ward",
    "ward_type": "ward_type", "zone": "ward_type",
    "category": "category", "type": "category", "water": "category",
    "road": "category", "roads": "category", "sanitation": "category",
    "status": "response_status", "response_status": "response_status",
    "response": "response_status", "rejected": "response_status",
}

METRIC_ALIASES = {
    "silence": "avg_silence", "score": "avg_silence", "silence score": "avg_silence",
    "avg": "avg_silence", "average": "avg_silence",
    "count": "count", "number": "count", "total": "count", "how many": "count",
    "percent": "silenced_pct", "percentage": "silenced_pct", "%": "silenced_pct",
    "silenced": "silenced_pct", "silenced_pct": "silenced_pct", "rate": "silenced_pct",
}

CHART_TYPE_ALIASES = {
    "bar": "bar", "bar chart": "bar", "vertical bar": "bar", "bars": "bar",
    "horizontal bar": "horizontalBar", "horizontal": "horizontalBar",
    "pie": "pie", "pie chart": "pie",
    "doughnut": "doughnut", "donut": "doughnut", "ring": "doughnut",
    "line": "line", "line chart": "line", "trend": "line",
    "radar": "radar", "spider": "radar", "web": "radar",
    "polar": "polarArea", "polar area": "polarArea",
    "stacked": "stacked", "stacked bar": "stacked",
    "grouped": "grouped", "grouped bar": "grouped",
    "heatmap": "heatmap", "hologram": "bar",
}


def _plan_chart_with_llm(message: str) -> Optional[Dict]:
    """
    Ask Gemini to parse user's chart request into a structured spec.
    Falls back to keyword parsing if LLM is unavailable.
    Returns {chart_type, fields[], metric, title} or None.
    """
    plan_prompt = f"""You are a data visualization planner. The user wants a chart from a civic complaints dataset.

AVAILABLE FIELDS: gender, caste, income_bracket, ward, ward_type, category, response_status
AVAILABLE METRICS: avg_silence (average silence score), count (number of complaints), silenced_pct (% silenced)
AVAILABLE CHART TYPES: bar, horizontalBar, pie, doughnut, line, radar, polarArea, stacked, grouped

User request: "{message}"

Respond ONLY with a JSON object (no markdown, no explanation):
{{
  "chart_type": "<type>",
  "fields": ["<field1>"] or ["<field1>", "<field2>"] for cross-tabulation,
  "metric": "<metric>",
  "title": "<short descriptive title>"
}}

Rules:
- If user mentions two demographic/categorical fields (e.g. "gender vs caste"), return both in fields[] for cross-tab.
- If user mentions one field, return a single-element fields[].
- If chart type is not clear, default to "bar".
- If metric is not clear, default to "avg_silence".
- For "pie" or "doughnut", prefer "count" metric unless user specifies otherwise.
- For temporal/time requests, use field "temporal" (special case)."""

    try:
        llm = get_llm()
        from langchain_core.messages import HumanMessage
        resp = llm.invoke([HumanMessage(content=plan_prompt)])
        text = resp.content.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        plan = json.loads(text)
        print(f"[CHART PLAN] LLM plan: {plan}")
        return plan
    except Exception as e:
        print(f"[CHART PLAN] LLM failed ({e}), using keyword fallback")
        return _plan_chart_keyword_fallback(message)


def _plan_chart_keyword_fallback(message: str) -> Optional[Dict]:
    """
    Keyword-based chart planning when LLM is unavailable.
    Parses chart type, fields, and metric from the message.
    """
    msg = message.lower()

    # ── Detect chart type ──
    chart_type = "bar"
    for phrase, ctype in sorted(CHART_TYPE_ALIASES.items(), key=lambda x: -len(x[0])):
        if phrase in msg:
            chart_type = ctype
            break

    # ── Detect fields (ordered by priority, longest match first) ──
    field_groups = {
        "gender": ["gender", "male", "female", "women", "men"],
        "caste": ["caste", "sc ", "st ", "obc", "general category"],
        "income_bracket": ["income", "poor", "rich", "wealth", "money"],
        "ward": ["ward"],
        "ward_type": ["ward_type", "zone", "elite", "slum"],
        "category": ["category", "water", "road", "sanitation", "electricity", "waste", "transport", "safety", "health"],
        "response_status": ["status", "response", "rejected", "no_response", "resolved"],
    }
    detected_fields = []
    for field, keywords in field_groups.items():
        if any(kw in msg for kw in keywords):
            if field not in detected_fields:
                detected_fields.append(field)

    # Check for "vs" / "versus" / "and" / "by" / "×" connecting two fields
    is_cross = any(w in msg for w in [' vs ', ' versus ', ' × ', ' x ', ' by ', ' across ', ' against '])

    if not detected_fields:
        # check for temporal
        if any(w in msg for w in ['time', 'temporal', 'trend', 'over time', 'month', 'decay']):
            detected_fields = ['temporal']
        else:
            detected_fields = ['gender']  # safe default

    # If only one field detected but user seems to want cross-tab, keep one field
    if len(detected_fields) == 1 and not is_cross:
        fields = detected_fields
    else:
        fields = detected_fields[:2]  # max 2

    # ── Detect metric ──
    metric = "avg_silence"
    for phrase, m in sorted(METRIC_ALIASES.items(), key=lambda x: -len(x[0])):
        if phrase in msg:
            metric = m
            break
    # Pie/doughnut default to count
    if chart_type in ('pie', 'doughnut', 'polarArea') and metric == 'avg_silence':
        metric = 'count'

    # ── Build title ──
    field_labels = [f.replace('_', ' ').title() for f in fields]
    if len(field_labels) == 2:
        title = f"{field_labels[0]} vs {field_labels[1]} — {metric.replace('_', ' ').title()}"
    else:
        title = f"{metric.replace('_', ' ').title()} by {field_labels[0]}"

    plan = {"chart_type": chart_type, "fields": fields, "metric": metric, "title": title}
    print(f"[CHART PLAN] Keyword fallback: {plan}")
    return plan


def _build_chart_from_plan(plan: Dict) -> Optional[Dict]:
    """
    Given a chart plan {chart_type, fields, metric, title}, fetch data
    and build a complete Chart.js config.
    """
    if not plan or 'fields' not in plan:
        return None

    chart_type = plan.get('chart_type', 'bar')
    fields = plan['fields']
    metric = plan.get('metric', 'avg_silence')
    title = plan.get('title', 'Analysis Chart')

    chart = {"type": chart_type, "data": {}, "options": {}, "title": title}

    # ── TEMPORAL special case ──
    if 'temporal' in fields:
        data = temporal_decay_analysis()
        chart["type"] = "line" if chart_type in ('bar', 'line') else chart_type
        chart["data"] = {
            "labels": [b["time_bucket"] for b in data],
            "datasets": [{
                "label": metric.replace('_', ' ').title(),
                "data": [b.get(metric, b.get('avg_silence', 0)) for b in data],
                "borderColor": CHART_COLORS[1],
                "backgroundColor": "rgba(255, 107, 107, 0.15)",
                "fill": True, "tension": 0.3
            }]
        }
        return chart

    # ── CROSS-TABULATION (2 fields) ──
    if len(fields) >= 2:
        ct = cross_tabulation(fields[0], fields[1], metric)
        labels_a = ct["labels_a"]  # x-axis groups
        labels_b = ct["labels_b"]  # datasets (one per label_b)

        datasets = []
        for j, lb in enumerate(labels_b):
            ds = {
                "label": lb,
                "data": [ct["matrix"][i][j] for i in range(len(labels_a))],
                "backgroundColor": CHART_COLORS[j % len(CHART_COLORS)],
            }
            if chart_type == 'line':
                ds["borderColor"] = CHART_COLORS[j % len(CHART_COLORS)]
                ds["fill"] = False
                ds["tension"] = 0.3
            datasets.append(ds)

        chart["data"] = {"labels": labels_a, "datasets": datasets}

        # Force grouped/stacked for multi-dataset bar
        if chart_type in ('bar', 'grouped', 'stacked', 'horizontalBar'):
            is_stacked = chart_type == 'stacked'
            chart["type"] = 'bar'
            chart["options"]["scales"] = {
                "x": {"stacked": is_stacked},
                "y": {"stacked": is_stacked}
            }
            if chart_type == 'horizontalBar':
                chart["options"]["indexAxis"] = "y"
        chart["options"]["_multiDataset"] = True
        return chart

    # ── SINGLE FIELD ──
    field = fields[0]
    top_n = 15 if field == 'ward' else 0
    data = field_breakdown(field, metric=metric, top_n=top_n)
    labels = [d['label'] for d in data]
    values = [d.get(metric, d.get('avg_silence', 0)) for d in data]
    bg_colors = [CHART_COLORS[i % len(CHART_COLORS)] for i in range(len(labels))]

    chart["data"] = {
        "labels": labels,
        "datasets": [{
            "label": metric.replace('_', ' ').title(),
            "data": values,
            "backgroundColor": bg_colors,
        }]
    }

    if chart_type == 'horizontalBar':
        chart["type"] = 'bar'
        chart["options"]["indexAxis"] = "y"
    elif chart_type == 'line':
        chart["data"]["datasets"][0]["borderColor"] = CHART_COLORS[0]
        chart["data"]["datasets"][0]["fill"] = False
        chart["data"]["datasets"][0]["tension"] = 0.3

    return chart


def _detect_chart_request(message: str) -> bool:
    """Quick check: does the user want a chart?"""
    msg = message.lower()
    return any(w in msg for w in [
        'chart', 'graph', 'plot', 'visualize', 'show me', 'display',
        'pie', 'bar', 'histogram', 'doughnut', 'donut', 'radar',
        'heatmap', 'hologram', 'polar', 'spider', 'stacked', 'grouped',
        'horizontal bar', 'vertical bar', 'line chart', 'trend chart',
        'draw', 'diagram', 'compare visually',
    ])

# TOOL REGISTRY

TOOLS = {
    "get_demographics": {
        "fn": tool_get_demographics,
        "desc": "Get silence breakdown by demographics (gender, caste, income)"
    },
    "get_geography": {
        "fn": tool_get_geography,
        "desc": "Get ward-level silence analysis"
    },
    "get_categories": {
        "fn": tool_get_categories,
        "desc": "Get silence breakdown by complaint category"
    },
    "get_temporal": {
        "fn": tool_get_temporal,
        "desc": "Get temporal decay analysis (how silence grows over time)"
    },
    "get_stats": {
        "fn": tool_get_stats,
        "desc": "Get overall statistics (total complaints, silence rate, etc.)"
    },
    "get_critical": {
        "fn": tool_get_critical,
        "desc": "Get critically silenced complaints (high silence scores)"
    },
    "search_complaints": {
        "fn": tool_search_complaints,
        "desc": "Semantic search for similar complaints"
    },
    "filtered_query": {
        "fn": tool_filtered_query,
        "desc": "Compound filtered query (e.g. water complaints in Ward 5 by caste)"
    },
}

# INTENT DETECTION — LLM-powered via Gemini function-calling
# Replaces old keyword-based intent with a smarter approach

def _detect_tools_from_message(message: str) -> tuple:
    """
    Detect which data tools to call and whether a chart is requested.
    Returns (tools_needed, wants_chart: bool).
    """
    msg = message.lower()
    tools_needed = []
    wants_chart = _detect_chart_request(message)

    # ── Compound/filtered query detection ──
    # If user combines specific filters (ward numbers, category names) with
    # a breakdown dimension, route to the filtered_query tool
    import re
    has_specific_ward = bool(re.search(r'ward\s*\d+', msg))
    has_specific_category = any(w in msg for w in [
        'water', 'road', 'roads', 'sanitation', 'electricity',
        'waste', 'transport', 'safety', 'health'
    ])
    has_breakdown_intent = any(p in msg for p in [
        ' by caste', ' by gender', ' by income', 'caste pattern',
        'gender pattern', 'income pattern', 'caste breakdown',
        'demographic', ' vs ward', 'compare ward', 'vs ward',
    ])
    has_comparison = any(w in msg for w in [' vs ', ' versus ', ' compare ', ' compared ', ' against '])

    is_compound = (
        (has_specific_ward and has_breakdown_intent) or
        (has_specific_category and has_breakdown_intent) or
        (has_specific_ward and has_specific_category) or
        (has_specific_ward and has_comparison)
    )

    if is_compound:
        tools_needed.append("filtered_query")

    # ── Standard data-tool detection ──
    if any(w in msg for w in ['gender', 'male', 'female', 'women', 'men', 'caste',
                               'income', 'demographic', 'sc', 'st', 'obc', 'poor', 'rich']):
        tools_needed.append("get_demographics")
    if any(w in msg for w in ['ward', 'area', 'location', 'geographic', 'region', 'where', 'zone']):
        tools_needed.append("get_geography")
    if any(w in msg for w in ['category', 'type', 'water', 'road', 'sanitation', 'electricity']):
        tools_needed.append("get_categories")
    if any(w in msg for w in ['time', 'temporal', 'trend', 'over time', 'days', 'waiting', 'decay', 'how long']):
        tools_needed.append("get_temporal")
    if any(w in msg for w in ['stat', 'overall', 'total', 'summary', 'overview', 'how many',
                               'silence rate', 'count']):
        tools_needed.append("get_stats")
    if any(w in msg for w in ['critical', 'worst', 'most silenced', 'highest score', 'neglected',
                               'extreme', 'urgent']):
        tools_needed.append("get_critical")
    if any(w in msg for w in ['search', 'find', 'look for', 'similar', 'like', 'about']):
        tools_needed.append("search_complaints")

    # Default
    if not tools_needed and not wants_chart:
        tools_needed.append("get_stats")

    return tools_needed, wants_chart


# ─── CONVERSATION MEMORY ───────────────────────────────────────

def _build_conversation_context(session_id: str, limit: int = 10) -> str:
    """Retrieve last N messages from session and format as conversation context."""
    history = get_session_history(session_id, limit=limit + 5)  # extra for system msgs
    # Filter to user/assistant only
    turns = [m for m in history if m.get("role") in ("user", "assistant")]
    # Take last `limit` turns
    turns = turns[-limit:]
    if not turns:
        return ""

    lines = ["CONVERSATION HISTORY:"]
    for m in turns:
        role_label = "USER" if m["role"] == "user" else "SENTINEL"
        # Truncate long assistant replies in history to save tokens
        content = m.get("content", "")
        if m["role"] == "assistant" and len(content) > 400:
            content = content[:400] + "…"
        lines.append(f"  {role_label}: {content}")
    return "\n".join(lines)

# MAIN CHAT FUNCTION

SYSTEM_PROMPT = """You are SENTINEL, an AI forensic analyst for the Silence Index project — a system that exposes institutional bias in civic complaint handling.

DATABASE: {total} civic complaints with silence scores (0-100, higher = more ignored)
DEMOGRAPHICS: gender (M/F/Other), caste (General/OBC/SC/ST), income (0-3L to 10L+)
LOCATION: 50 municipal wards across Elite, Middle, Poor, and Slum zones
CATEGORIES: Water, Roads, Sanitation, Electricity, Waste, Transport, Safety, Health
STATUSES: NO_RESPONSE, RESPONDED, RESOLVED, REJECTED

YOUR ROLE:
1. Analyze bias patterns with precision — cite specific numbers & percentages.
2. Compare privileged vs marginalized groups with disparity ratios.
3. Highlight anomalies, red flags, and systemic issues.
4. If the user asks a follow-up that references earlier data, use the conversation history.
5. When mentioning images, note multimodal complaints can be submitted via the upload button.

RESPONSE STYLE:
- Start with key finding or verdict
- Support with data points
- Use bullet points and markdown formatting
- Keep responses under 200 words unless detailed analysis requested
- Never apologize — be direct and forensic"""


def chat(message: str, session_id: str) -> Dict[str, Any]:
    """
    Process chat message with conversation memory.
    Steps:
      1. Store user message
      2. Retrieve last 10 conversation turns for context
      3. Detect intent & gather data
      4. Build prompt with history + data
      5. Call LLM
      6. Store & return response
    """
    # 1. Store user message
    store_message(session_id, "user", message)

    # 2. Conversation memory
    conversation_context = _build_conversation_context(session_id, limit=10)

    # 3. Detect tools & chart intent
    tools_needed, wants_chart = _detect_tools_from_message(message)

    # 4. Execute tools
    context_parts = []
    for tool_name in tools_needed:
        if tool_name in TOOLS:
            try:
                if tool_name == "search_complaints":
                    result = TOOLS[tool_name]["fn"](query=message)
                elif tool_name == "filtered_query":
                    result = TOOLS[tool_name]["fn"](message=message)
                else:
                    result = TOOLS[tool_name]["fn"]()
                context_parts.append(f"[{tool_name}]\n{json.dumps(result, indent=2, default=str)}")
            except Exception as e:
                context_parts.append(f"[{tool_name}] Error: {str(e)}")

    # 5. Chart — LLM plans, then we build from real data
    chart_data = None
    if wants_chart:
        try:
            plan = _plan_chart_with_llm(message)
            if plan:
                chart_data = _build_chart_from_plan(plan)
        except Exception as e:
            print(f"[CHART] Error: {e}")

    # 6. Build prompt with memory
    data_context = "\n\n".join(context_parts) if context_parts else "No specific data requested."
    total = get_total_complaints()

    full_prompt = f"""{SYSTEM_PROMPT.format(total=total)}

{conversation_context}

DATA RETRIEVED:
{data_context}

USER QUERY: {message}

Provide analysis based on the data above. Be specific with numbers. Use markdown."""

    # 7. LLM call
    try:
        llm = get_llm()
        from langchain_core.messages import HumanMessage
        response = llm.invoke([HumanMessage(content=full_prompt)])
        response_text = response.content
    except Exception as e:
        response_text = f"Analysis error: {str(e)}"

    # 8. Store AI response
    store_message(
        session_id,
        "assistant",
        response_text,
        metadata={
            "tools_used": tools_needed,
            "chart_generated": chart_data is not None,
        },
    )

    return {
        "response": response_text,
        "chart": chart_data,
        "tools_used": tools_needed,
    }

# TEST

if __name__ == "__main__":
    print("Testing Chat Agent...")
    
    # Create session
    session = create_session("Test Investigation")
    print(f"Session: {session}")
    
    # Test chat
    result = chat("What's the overall silence rate?", session)
    print(f"Response: {result['response'][:200]}...")
    print(f"Tools: {result['tools_used']}")
    
    # Test history
    history = get_session_history(session)
    print(f"Messages in session: {len(history)}")
