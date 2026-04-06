"""Pinecone vector ingestion and upsertion."""

import logging
import os
import time
from typing import Any

from pinecone import Pinecone

logger = logging.getLogger(__name__)

BATCH_SIZE = 100
MAX_RETRIES = 2
RETRY_DELAY = 2.0  # seconds


def get_pinecone_index(index_name: str | None = None):
    """
    Initialize Pinecone client and get index.

    Args:
        index_name: Pinecone index name. If None, reads from PINECONE_INDEX_NAME env var.

    Returns:
        Pinecone Index object
    """
    if index_name is None:
        index_name = os.getenv("PINECONE_INDEX_NAME")
        if not index_name:
            raise ValueError("PINECONE_INDEX_NAME environment variable not set")

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable not set")

    logger.info(f"Connecting to Pinecone index: {index_name}")

    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    return index


def chunk_vectors(
    vectors: list[tuple[str, list[float], dict[str, Any]]], batch_size: int = BATCH_SIZE
):
    """
    Chunk vectors into batches for upsert.

    Yields batches of vectors (tuples of id, embedding, metadata).
    """
    for i in range(0, len(vectors), batch_size):
        yield vectors[i : i + batch_size]


def upsert_batch(index, batch: list[tuple[str, list[float], dict[str, Any]]], batch_num: int) -> bool:
    """
    Upsert a single batch to Pinecone with retry logic.

    Args:
        index: Pinecone Index object
        batch: List of (vector_id, embedding, metadata) tuples
        batch_num: Batch number for logging

    Returns:
        True if successful, False if failed after retries
    """
    vectors_to_upsert = [
        {"id": vec_id, "values": embedding, "metadata": metadata}
        for vec_id, embedding, metadata in batch
    ]

    for attempt in range(MAX_RETRIES + 1):
        try:
            logger.debug(f"Upserting batch {batch_num} ({len(batch)} vectors), attempt {attempt + 1}")

            # Use default namespace (ISO date from metadata could be added here)
            index.upsert(vectors=vectors_to_upsert, namespace="default")

            logger.info(f"✓ Batch {batch_num}: upserted {len(batch)} vectors")
            return True

        except Exception as e:
            if attempt < MAX_RETRIES:
                delay = RETRY_DELAY * (2 ** attempt)  # exponential backoff
                logger.warning(
                    f"Batch {batch_num} upsert failed (attempt {attempt + 1}/{MAX_RETRIES + 1}): {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
            else:
                logger.error(f"✗ Batch {batch_num} failed after {MAX_RETRIES + 1} attempts: {e}")
                return False

    return False


def ingest_vectors(
    vectors: list[tuple[str, list[float], dict[str, Any]]], index_name: str | None = None
) -> dict[str, Any]:
    """
    Ingest a list of vectors into Pinecone in batches.

    Args:
        vectors: List of (vector_id, embedding, metadata) tuples
        index_name: Pinecone index name (optional, read from env if not provided)

    Returns:
        Summary dict with success/fail counts
    """
    index = get_pinecone_index(index_name)

    total_vectors = len(vectors)
    logger.info(f"Starting ingestion of {total_vectors} vectors...")

    successful_batches = 0
    failed_batches = 0
    total_upserted = 0

    for batch_num, batch in enumerate(chunk_vectors(vectors, BATCH_SIZE), start=1):
        if upsert_batch(index, batch, batch_num):
            successful_batches += 1
            total_upserted += len(batch)
        else:
            failed_batches += 1

    # Summary
    summary = {
        "total_vectors": total_vectors,
        "total_upserted": total_upserted,
        "successful_batches": successful_batches,
        "failed_batches": failed_batches,
        "success": failed_batches == 0,
    }

    logger.info(
        f"Ingestion complete. "
        f"Upserted: {total_upserted}/{total_vectors} vectors. "
        f"Batches: {successful_batches} successful, {failed_batches} failed."
    )

    return summary
