import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

load_dotenv()

# Qdrant client
qdrant = QdrantClient(host="localhost", port=6333)

# Chat history collection name
CHAT_COLLECTION = "chat_history"

# Lazy loading for models
_llm = None
_embedding_model = None

def get_llm():
    """Lazy load the LLM"""
    global _llm
    if _llm is None:
        from langchain_google_genai import ChatGoogleGenerativeAI
        _llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv('GEMINI_API_KEY'),
            temperature=0.7
        )
    return _llm

def get_embedding_model():
    """Lazy load embedding model"""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    return _embedding_model

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
    get_silenced_complaints
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
    total = 10000
    
    avg_silence = sum(c['silence_score'] for c in silenced) / len(silenced) if silenced else 0
    avg_days = sum(c['days_in_system'] for c in silenced) / len(silenced) if silenced else 0
    
    return {
        "total_complaints": total,
        "total_silenced": total_silenced,
        "silence_rate": round((total_silenced / total) * 100, 1),
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

def tool_search_complaints(query: str, limit: int = 10) -> Dict:
    """Semantic search for complaints"""
    results = similarity_search(query, limit=limit)
    return {"query": query, "total": len(results), "results": results}

def tool_generate_chart(chart_type: str, data_source: str) -> Dict:
    """Generate Chart.js config for visualization"""
    chart = {"type": chart_type, "data": {}, "options": {}}
    
    if data_source == "gender":
        data = demographic_breakdown()["by_gender"]
        chart["data"] = {
            "labels": list(data.keys()),
            "datasets": [{
                "label": "Avg Silence Score",
                "data": [v["avg_silence"] for v in data.values()],
                "backgroundColor": ["#00d4ff", "#ff6b6b", "#ffb700"]
            }]
        }
        chart["title"] = "Silence Score by Gender"
        
    elif data_source == "caste":
        data = demographic_breakdown()["by_caste"]
        chart["data"] = {
            "labels": list(data.keys()),
            "datasets": [{
                "label": "Avg Silence Score",
                "data": [v["avg_silence"] for v in data.values()],
                "backgroundColor": ["#00d4ff", "#00ff88", "#ffb700", "#ff6b6b"]
            }]
        }
        chart["title"] = "Silence Score by Caste"
        
    elif data_source == "income":
        data = demographic_breakdown()["by_income"]
        order = ['0-3L', '3-6L', '6-10L', '10L+']
        chart["data"] = {
            "labels": order,
            "datasets": [{
                "label": "Avg Silence Score",
                "data": [data.get(i, {}).get("avg_silence", 0) for i in order],
                "backgroundColor": ["#ff6b6b", "#ffb700", "#00d4ff", "#00ff88"]
            }]
        }
        chart["title"] = "Silence Score by Income Level"
        
    elif data_source == "geographic":
        data = geographic_breakdown()["top_silenced"][:10]
        chart["data"] = {
            "labels": [w["ward"] for w in data],
            "datasets": [{
                "label": "Avg Silence Score",
                "data": [w["avg_silence"] for w in data],
                "backgroundColor": "#ff6b6b"
            }]
        }
        chart["title"] = "Top 10 Silenced Wards"
        
    elif data_source == "categories":
        data = complaint_type_analysis()
        chart["type"] = "bar"
        chart["data"] = {
            "labels": [c["category"] for c in data],
            "datasets": [{
                "label": "Silenced %",
                "data": [c["silenced_pct"] for c in data],
                "backgroundColor": "#ff6b6b"
            }]
        }
        chart["title"] = "Silenced % by Category"
        chart["options"]["indexAxis"] = "y"
        
    elif data_source == "temporal":
        data = temporal_decay_analysis()
        chart["type"] = "line"
        chart["data"] = {
            "labels": [b["time_bucket"] for b in data],
            "datasets": [{
                "label": "Silenced %",
                "data": [b["silenced_pct"] for b in data],
                "borderColor": "#ff6b6b",
                "backgroundColor": "rgba(255, 107, 107, 0.1)",
                "fill": True,
                "tension": 0.3
            }]
        }
        chart["title"] = "Silence Growth Over Time"
    
    return chart

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
    "generate_chart": {
        "fn": tool_generate_chart,
        "desc": "Generate chart visualization (gender/caste/income/geographic/categories/temporal)"
    }
}

# INTENT DETECTION & TOOL ROUTING

def detect_intent(message: str) -> tuple:
    """Detect which tools to call based on message"""
    msg = message.lower()
    tools_needed = []
    chart_request = None
    
    # Chart detection
    if any(w in msg for w in ['chart', 'graph', 'plot', 'visualize', 'show me', 'display']):
        if any(w in msg for w in ['gender', 'male', 'female', 'women', 'men']):
            chart_request = ("bar", "gender")
        elif any(w in msg for w in ['caste', 'sc', 'st', 'obc', 'general']):
            chart_request = ("bar", "caste")
        elif any(w in msg for w in ['income', 'poor', 'rich', 'wealth', 'money']):
            chart_request = ("bar", "income")
        elif any(w in msg for w in ['ward', 'area', 'location', 'geographic', 'region']):
            chart_request = ("bar", "geographic")
        elif any(w in msg for w in ['category', 'type', 'water', 'road', 'sanitation']):
            chart_request = ("bar", "categories")
        elif any(w in msg for w in ['time', 'temporal', 'trend', 'over time', 'decay']):
            chart_request = ("line", "temporal")
    
    # Tool detection
    if any(w in msg for w in ['gender', 'male', 'female', 'women', 'men', 'caste', 'income', 'demographic', 'sc', 'st', 'obc']):
        tools_needed.append("get_demographics")
    
    if any(w in msg for w in ['ward', 'area', 'location', 'geographic', 'region', 'where']):
        tools_needed.append("get_geography")
    
    if any(w in msg for w in ['category', 'type', 'water', 'road', 'sanitation', 'electricity']):
        tools_needed.append("get_categories")
    
    if any(w in msg for w in ['time', 'temporal', 'trend', 'over time', 'days', 'waiting', 'decay']):
        tools_needed.append("get_temporal")
    
    if any(w in msg for w in ['stat', 'overall', 'total', 'summary', 'overview', 'how many']):
        tools_needed.append("get_stats")
    
    if any(w in msg for w in ['critical', 'worst', 'most silenced', 'highest score', 'neglected']):
        tools_needed.append("get_critical")
    
    if any(w in msg for w in ['search', 'find', 'look for', 'similar', 'like']):
        tools_needed.append("search_complaints")
    
    # Default to stats if nothing detected
    if not tools_needed and not chart_request:
        tools_needed.append("get_stats")
    
    return tools_needed, chart_request

# MAIN CHAT FUNCTION

SYSTEM_PROMPT = """You are SENTINEL, an AI forensic analyst for the Silence Index project - a system that exposes institutional bias in civic complaint handling.

DATABASE: 10,000 civic complaints with silence scores (0-100, higher = more ignored)
DEMOGRAPHICS: gender (M/F/Other), caste (General/OBC/SC/ST), income (0-3L to 10L+)
LOCATION: 50 municipal wards
CATEGORIES: Water, Roads, Sanitation, Electricity, Waste, Transport, Safety, Health

YOUR ROLE:
1. Analyze bias patterns with precision
2. Report findings with specific numbers and percentages
3. Highlight disparities and systemic issues
4. Be direct, concise, and data-driven
5. Use forensic/investigative tone

RESPONSE STYLE:
- Start with key finding or verdict
- Support with data points
- Use bullet points for clarity
- Highlight anomalies and red flags
- Keep responses under 200 words unless detailed analysis requested"""

def chat(message: str, session_id: str) -> Dict[str, Any]:
    """
    Process chat message and return response with optional chart.
    Stores both user message and AI response in Qdrant.
    """
    # Store user message
    store_message(session_id, "user", message)
    
    # Detect intent and gather data
    tools_needed, chart_request = detect_intent(message)
    
    # Execute tools
    context_parts = []
    for tool_name in tools_needed:
        if tool_name in TOOLS:
            try:
                result = TOOLS[tool_name]["fn"]()
                context_parts.append(f"[{tool_name}]\n{json.dumps(result, indent=2)}")
            except Exception as e:
                context_parts.append(f"[{tool_name}] Error: {str(e)}")
    
    # Generate chart if requested
    chart_data = None
    if chart_request:
        chart_type, data_source = chart_request
        chart_data = tool_generate_chart(chart_type, data_source)
    
    # Build prompt
    data_context = "\n\n".join(context_parts) if context_parts else "No specific data requested."
    
    full_prompt = f"""{SYSTEM_PROMPT}

DATA RETRIEVED:
{data_context}

USER QUERY: {message}

Provide analysis based on the data above. Be specific with numbers."""
    
    # Get LLM response
    try:
        llm = get_llm()
        from langchain_core.messages import HumanMessage
        response = llm.invoke([HumanMessage(content=full_prompt)])
        response_text = response.content
    except Exception as e:
        response_text = f"Analysis error: {str(e)}"
    
    # Store AI response
    store_message(
        session_id, 
        "assistant", 
        response_text,
        metadata={
            "tools_used": tools_needed,
            "chart_generated": chart_data is not None
        }
    )
    
    return {
        "response": response_text,
        "chart": chart_data,
        "tools_used": tools_needed
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
