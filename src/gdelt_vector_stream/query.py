"""Semantic search interface for querying Pinecone index."""

import logging
import os
from typing import Any

from dotenv import load_dotenv

from gdelt_vector_stream.embedder import embed_event, get_embedder
from gdelt_vector_stream.ingestor import get_pinecone_index

# Load environment variables from .env
load_dotenv()

logger = logging.getLogger(__name__)


def semantic_search(query_text: str, index_name: str | None = None, top_k: int = 5) -> list[dict[str, Any]]:
    """
    Semantic search over the GDELT events index.

    Args:
        query_text: Natural language query (e.g., "protests in Asia")
        index_name: Pinecone index name (optional, read from env if not provided)
        top_k: Number of top results to return

    Returns:
        List of result dicts with event metadata and similarity score
    """
    logger.info(f"Searching for: {query_text}")

    # Get embedder and index
    embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embedder = get_embedder(embedding_model)
    index = get_pinecone_index(index_name)

    # Embed query
    query_embedding = embed_event(query_text, embedder)

    # Search Pinecone
    results = index.query(
        vector=query_embedding, top_k=top_k, include_metadata=True, namespace="default"
    )

    # Format results
    formatted = []
    for match in results.get("matches", []):
        result = {
            "vector_id": match["id"],
            "similarity_score": match.get("score", 0),
            "metadata": match.get("metadata", {}),
        }
        formatted.append(result)
        logger.info(
            f"  [{result['similarity_score']:.3f}] {result['metadata'].get('actor1_name', 'Unknown')} "
            f"in {result['metadata'].get('country_code', '?')}"
        )

    return formatted


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m gdelt_vector_stream.query '<query_text>'")
        print("Example: python -m gdelt_vector_stream.query 'protests in Asia'")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    results = semantic_search(query)

    print(f"\nFound {len(results)} results:")
    for i, result in enumerate(results, 1):
        meta = result["metadata"]
        print(
            f"\n{i}. [{result['similarity_score']:.3f}] "
            f"{meta.get('actor1_name', '?')} → {meta.get('actor2_name', '?')} "
            f"in {meta.get('country_code', '?')}"
        )
        print(f"   URL: {meta.get('source_url', 'N/A')}")
