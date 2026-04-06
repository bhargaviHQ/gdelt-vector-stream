"""Main orchestration: fetch → embed → ingest."""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from gdelt_vector_stream.embedder import create_pinecone_vectors, embed_events_batch, get_embedder
from gdelt_vector_stream.fetcher import create_event_text, load_gdelt_events
from gdelt_vector_stream.ingestor import ingest_vectors

# Load environment variables from .env
load_dotenv()

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def ingest_gdelt_csv(csv_path: str | Path) -> dict:
    """
    End-to-end ingestion pipeline: load CSV → embed → upsert to Pinecone.

    Args:
        csv_path: Path to GDELT 2.0 Event CSV file

    Returns:
        Ingestion summary dict
    """
    csv_path = Path(csv_path)

    logger.info(f"Starting ingestion pipeline for {csv_path}")

    # Step 1: Fetch and parse
    logger.info("Step 1/3: Loading GDELT events...")
    events = load_gdelt_events(csv_path)

    if not events:
        logger.warning("No events loaded. Exiting.")
        return {"total_vectors": 0, "success": False}

    # Step 2: Create text summaries and embed
    logger.info("Step 2/3: Creating event texts and embeddings...")
    event_texts = [create_event_text(event) for event in events]

    # Get embedder (lazy-loads model on first call)
    embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embedder = get_embedder(embedding_model)

    # Embed all events
    embeddings = embed_events_batch(event_texts, embedder)

    # Step 3: Create Pinecone vectors and ingest
    logger.info("Step 3/3: Creating Pinecone vectors and upserting...")
    vectors = create_pinecone_vectors(events, event_texts, embeddings)

    summary = ingest_vectors(vectors)

    return summary


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m gdelt_vector_stream.main <csv_path>")
        print("Example: python -m gdelt_vector_stream.main data/sample_gdelt_events.csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    result = ingest_gdelt_csv(csv_path)

    if result["success"]:
        logger.info("✓ Ingestion successful!")
        sys.exit(0)
    else:
        logger.error("✗ Ingestion failed!")
        sys.exit(1)
