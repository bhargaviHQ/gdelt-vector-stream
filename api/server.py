"""FastAPI backend for gdelt-vector-stream."""

import logging
import os

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import InferenceClient
from pydantic import BaseModel

from gdelt_vector_stream.analyst import ask as analyst_ask
from gdelt_vector_stream.downloader import download_and_ingest, load_processed
from gdelt_vector_stream.ingestor import get_pinecone_index
from gdelt_vector_stream.query import semantic_search
from gdelt_vector_stream.trends import DEFAULT_CATEGORIES, get_trends_digest

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
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

HF_TOKEN = os.getenv("HF_TOKEN")


# --- Request/Response Models ---

HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")


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
    """Check connectivity to Pinecone and Hugging Face Inference API."""
    pinecone_ok = False
    hf_ok = False

    try:
        index = get_pinecone_index()
        index.describe_index_stats()
        pinecone_ok = True
    except Exception as e:
        logger.warning(f"Pinecone health check failed: {e}")

    try:
        if HF_TOKEN:
            client = InferenceClient(token=HF_TOKEN)
            client.get_model_status(HF_MODEL)
            hf_ok = True
    except Exception as e:
        logger.warning(f"HF Inference health check failed: {e}")

    if pinecone_ok and hf_ok:
        status = "ok"
    elif pinecone_ok:
        status = "degraded"
    else:
        status = "error"

    return {
        "status": status,
        "pinecone_connected": pinecone_ok,
        "hf_available": hf_ok,
        "hf_model": HF_MODEL,
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
        logger.error(f"Failed to get stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve index statistics")


@app.get("/api/search")
def search(q: str = Query(..., min_length=1), top_k: int = Query(5, ge=1, le=20)):
    """Semantic search over GDELT events."""
    try:
        results = semantic_search(query_text=q, top_k=top_k)
        return {"query": q, "top_k": top_k, "results": results}
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Search failed")


@app.post("/api/ask")
def ask(req: AskRequest):
    """Ask a question — RAG answer grounded in GDELT events via Hugging Face Inference API."""
    try:
        result = analyst_ask(question=req.question, top_k=req.top_k, model=req.model)
        return result
    except RuntimeError as e:
        return {
            "answer": None,
            "events": [],
            "model": req.model or HF_MODEL,
            "error": "hf_error",
            "message": str(e),
        }
    except Exception as e:
        logger.error(f"Analyst failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Analyst failed")


def _run_ingest(sample_size: int, max_files: int) -> None:
    """Background worker that runs the GDELT ingestion pipeline."""
    try:
        download_and_ingest(sample_size=sample_size, max_files=max_files)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


@app.get("/api/trends")
def get_trending_digest(
    top_k: int = Query(3, ge=1, le=10),
    categories: list[str] = Query(default=None),
    model: str = Query(default=None),
):
    """Generate a World News Digest across broad topic categories."""
    try:
        result = get_trends_digest(
            categories=categories or DEFAULT_CATEGORIES,
            top_k=top_k,
            model=model,
        )
        return result
    except RuntimeError as e:
        return {
            "digest": None,
            "categories": {},
            "model": model or HF_MODEL,
            "total_events": 0,
            "error": "hf_error",
            "message": str(e),
        }
    except Exception:
        logger.exception("Trends digest failed")
        raise HTTPException(status_code=500, detail="Trends digest failed. Check server logs.")
