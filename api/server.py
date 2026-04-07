"""FastAPI backend for gdelt-vector-stream."""

import logging
import os

import requests as http_requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from gdelt_vector_stream.analyst import ask as analyst_ask
from gdelt_vector_stream.downloader import download_and_ingest, load_processed
from gdelt_vector_stream.ingestor import get_pinecone_index
from gdelt_vector_stream.query import semantic_search

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="GDELT Vector Stream API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


# --- Request/Response Models ---

class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    model: str | None = None


class IngestRequest(BaseModel):
    sample_size: int = 100
    max_files: int = 1


# --- Endpoints ---

@app.get("/api/health")
def health():
    """Check connectivity to Pinecone and Ollama."""
    pinecone_ok = False
    ollama_ok = False

    try:
        index = get_pinecone_index()
        index.describe_index_stats()
        pinecone_ok = True
    except Exception as e:
        logger.warning(f"Pinecone health check failed: {e}")

    try:
        resp = http_requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        ollama_ok = resp.status_code == 200
    except Exception:
        pass

    if pinecone_ok and ollama_ok:
        status = "ok"
    elif pinecone_ok:
        status = "degraded"
    else:
        status = "error"

    return {
        "status": status,
        "pinecone_connected": pinecone_ok,
        "ollama_available": ollama_ok,
        "ollama_url": OLLAMA_BASE_URL,
    }


@app.get("/api/stats")
def stats():
    """Get index statistics."""
    try:
        index = get_pinecone_index()
        index_stats = index.describe_index_stats()

        # Handle both dict and object responses
        if hasattr(index_stats, "total_vector_count"):
            vector_count = index_stats.total_vector_count
            namespaces = index_stats.namespaces or {}
            fullness = index_stats.index_fullness
        else:
            vector_count = index_stats.get("total_vector_count", 0)
            namespaces = index_stats.get("namespaces", {})
            fullness = index_stats.get("index_fullness", 0)

        processed = load_processed()

        return {
            "vector_count": vector_count,
            "namespace_count": len(namespaces),
            "processed_files": len(processed),
            "index_fullness": fullness,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {e}")


@app.get("/api/search")
def search(q: str = Query(..., min_length=1), top_k: int = Query(5, ge=1, le=20)):
    """Semantic search over GDELT events."""
    try:
        results = semantic_search(query_text=q, top_k=top_k)
        return {"query": q, "top_k": top_k, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")


@app.post("/api/ask")
def ask(req: AskRequest):
    """Ask a question — RAG answer grounded in GDELT events via Ollama."""
    try:
        result = analyst_ask(question=req.question, top_k=req.top_k, model=req.model)
        return result
    except ConnectionError as e:
        return {
            "answer": None,
            "events": [],
            "model": req.model or os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
            "error": "ollama_unavailable",
            "message": str(e),
        }
    except RuntimeError as e:
        return {
            "answer": None,
            "events": [],
            "model": req.model,
            "error": "model_not_found",
            "message": str(e),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analyst failed: {e}")


@app.post("/api/ingest")
def ingest(req: IngestRequest):
    """Trigger GDELT data ingestion."""
    try:
        summaries = download_and_ingest(
            sample_size=req.sample_size, max_files=req.max_files
        )
        return {"summaries": summaries, "files_processed": len(summaries)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")
