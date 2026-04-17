"""Embedding generation for GDELT events."""

import logging
import math
from typing import Any

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Lazy-loaded models, keyed by model name so different model names are cached separately
_models: dict[str, SentenceTransformer] = {}


def get_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Get or initialize the embedding model (lazy loading).

    The model is cached by name, so requesting a different model name returns
    the correct model rather than reusing the first-loaded one.

    Args:
        model_name: Hugging Face model identifier

    Returns:
        SentenceTransformer model instance
    """
    if model_name not in _models:
        logger.info(f"Loading embedding model: {model_name}")
        _models[model_name] = SentenceTransformer(model_name)
        logger.info(f"Model loaded. Embedding dimension: {_models[model_name].get_sentence_embedding_dimension()}")

    return _models[model_name]


def embed_event(event_text: str, embedder: SentenceTransformer) -> list[float]:
    """
    Generate embedding for a single event text.

    Args:
        event_text: Text summary of the event (from fetcher.create_event_text)
        embedder: SentenceTransformer instance

    Returns:
        Embedding vector (list of floats)
    """
    embedding = embedder.encode(event_text, convert_to_numpy=True)
    if hasattr(embedding, 'tolist'):
        return embedding.tolist()
    return embedding


def embed_events_batch(
    texts: list[str], embedder: SentenceTransformer, show_progress_bar: bool = False
) -> list[list[float]]:
    """
    Generate embeddings for multiple event texts in a batch.

    Args:
        texts: List of event text summaries
        embedder: SentenceTransformer instance
        show_progress_bar: Whether to show a progress bar

    Returns:
        List of embedding vectors
    """
    logger.debug(f"Embedding batch of {len(texts)} events")

    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=show_progress_bar)

    # Convert numpy array to list of lists
    if hasattr(embeddings, 'tolist'):
        return embeddings.tolist()
    return embeddings


def _safe_string(value: Any) -> str:
    """Safely convert value to string, handling NaN and None."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def create_pinecone_vectors(
    events: list[dict[str, Any]], event_texts: list[str], embeddings: list[list[float]]
) -> list[tuple[str, list[float], dict[str, Any]]]:
    """
    Create Pinecone-formatted vectors from events, texts, and embeddings.

    Returns list of tuples: (vector_id, vector, metadata)

    Vector ID format: gdelt-{YYYYMMDDHHMMSS}-{event_id}
    """
    vectors = []

    for event, text, embedding in zip(events, event_texts, embeddings):
        # Vector ID: gdelt-{SQLDATE}-{GLOBALEVENTID}
        sql_date = str(event.get("SQLDATE", ""))
        event_id = str(event.get("GLOBALEVENTID", "")).zfill(10)
        vector_id = f"gdelt-{sql_date}-{event_id}"

        # Lean metadata schema (per VectorIngestion.md)
        metadata = {
            "event_date": f"{sql_date[:4]}-{sql_date[4:6]}-{sql_date[6:8]}",  # Convert YYYYMMDD to YYYY-MM-DD
            "country_code": _safe_string(event.get("ActionGeo_CountryCode", "")),
            "event_code": _safe_string(event.get("EventCode", "")),
            "event_base_code": _safe_string(event.get("EventBaseCode", "")),
            "actor1_name": _safe_string(event.get("Actor1Name", "")),
            "actor2_name": _safe_string(event.get("Actor2Name", "")),
            "avg_tone": float(event.get("AvgTone", 0)),
            "num_mentions": int(event.get("NumMentions", 0)),
            "source_url": _safe_string(event.get("SourceURL", "")),
        }

        vectors.append((vector_id, embedding, metadata))

    return vectors
