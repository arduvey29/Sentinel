"""
Shared model instances for the Sentinel application.
Ensures a single LLM and embedding model are reused across all modules.
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

# ── Qdrant Client (singleton) ──────────────────────────────────
qdrant = QdrantClient(host="localhost", port=6333)

# ── Collection names ────────────────────────────────────────────
COMPLAINTS_COLLECTION = "silence_complaints"
CHAT_COLLECTION = "chat_history"

# ── Lazy-loaded singletons ─────────────────────────────────────
_llm = None
_embedding_model = None


def get_llm(temperature: float = 0.7):
    """Return the shared Gemini LLM instance (lazy-loaded)."""
    global _llm
    if _llm is None:
        from langchain_google_genai import ChatGoogleGenerativeAI
        _llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=temperature,
        )
        print("[models] Gemini 2.5 Flash loaded")
    return _llm


def get_embedding_model():
    """Return the shared SentenceTransformer instance (lazy-loaded)."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        print("[models] BAAI/bge-small-en-v1.5 loaded (384-dim)")
    return _embedding_model


def get_total_complaints() -> int:
    """Get actual complaint count from Qdrant instead of hardcoding."""
    try:
        info = qdrant.get_collection(COMPLAINTS_COLLECTION)
        return info.points_count
    except Exception:
        return 10000  # fallback
