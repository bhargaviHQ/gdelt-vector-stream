"""Trending Topics Digest: surface the top global news themes from your GDELT index."""

import argparse
import logging
import os
import sys
from typing import Any

from dotenv import load_dotenv

from gdelt_vector_stream.analyst import call_hf_inference, format_events_as_context
from gdelt_vector_stream.query import semantic_search

load_dotenv()

logger = logging.getLogger(__name__)

HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")

# Default broad CAMEO-inspired topic categories used for trend discovery
DEFAULT_CATEGORIES: list[str] = [
    "military conflict and armed violence",
    "diplomacy and international cooperation",
    "protests and civil unrest",
    "economic policy and financial markets",
    "climate change and environmental disasters",
    "public health and disease outbreaks",
    "humanitarian crisis and refugee movements",
    "technology and cybersecurity",
]


def get_trending_events(
    categories: list[str],
    top_k: int = 3,
) -> dict[str, list[dict[str, Any]]]:
    """
    Retrieve the most relevant GDELT events for each topic category.

    Runs one semantic search per category and deduplicates events globally so
    the same vector never appears under more than one category.

    Args:
        categories: List of topic strings used as search queries.
        top_k: Number of events to retrieve per category.

    Returns:
        Ordered dict mapping each category label to its list of event dicts.
    """
    seen_ids: set[str] = set()
    category_events: dict[str, list[dict[str, Any]]] = {}

    for category in categories:
        logger.info(f"Searching for category: {category!r}")
        results = semantic_search(category, top_k=top_k)

        # Keep only events not already claimed by an earlier category
        unique_results = []
        for event in results:
            vid = event["vector_id"]
            if vid not in seen_ids:
                seen_ids.add(vid)
                unique_results.append(event)

        category_events[category] = unique_results
        logger.info(f"  {len(unique_results)} unique events for {category!r}")

    return category_events


def build_trends_prompt(category_events: dict[str, list[dict[str, Any]]]) -> str:
    """
    Build the LLM prompt for a multi-category world news digest.

    Args:
        category_events: Mapping of category label → list of event dicts.

    Returns:
        Full prompt string ready to send to the Inference API.
    """
    sections: list[str] = []
    for category, events in category_events.items():
        if not events:
            continue
        context = format_events_as_context(events)
        sections.append(f"### {category.capitalize()}\n{context}")

    all_context = "\n\n".join(sections)

    return f"""You are a geopolitical news analyst producing a World News Digest. \
Using ONLY the GDELT news events below, write one concise paragraph per topic category \
summarising what is happening. Skip any category that has no relevant events. \
End with a "Sources" section listing all unique source URLs referenced.

--- GDELT NEWS EVENTS BY CATEGORY ---
{all_context}
--- END EVENTS ---

World News Digest:"""


def get_trends_digest(
    categories: list[str] | None = None,
    top_k: int = 3,
    model: str | None = None,
) -> dict[str, Any]:
    """
    Generate a World News Digest from the GDELT vector index.

    Args:
        categories: Topic categories to search (defaults to DEFAULT_CATEGORIES).
        top_k: Number of events to retrieve per category.
        model: HF model override (reads HF_MODEL env var if not given).

    Returns:
        Dict with keys:
          - ``digest``: LLM-generated narrative summary string.
          - ``categories``: Mapping of category → list of retrieved events.
          - ``model``: Model ID used.
          - ``total_events``: Total unique events retrieved.
    """
    categories = categories or DEFAULT_CATEGORIES
    model = model or HF_MODEL

    # Step 1: retrieve events per category
    category_events = get_trending_events(categories, top_k=top_k)

    total_events = sum(len(evts) for evts in category_events.values())
    logger.info(f"Retrieved {total_events} unique events across {len(categories)} categories")

    if total_events == 0:
        return {
            "digest": "No events found in the GDELT index. Try ingesting more data first.",
            "categories": {cat: [] for cat in categories},
            "model": model,
            "total_events": 0,
        }

    # Step 2: build prompt and call LLM
    prompt = build_trends_prompt(category_events)
    logger.info(f"Sending digest prompt to {model}...")
    digest = call_hf_inference(prompt, model=model)

    return {
        "digest": digest,
        "categories": category_events,
        "model": model,
        "total_events": total_events,
    }


# --- CLI ---

if __name__ == "__main__":
    log_level = os.getenv("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Generate a World News Digest from your GDELT vector index"
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Override the default topic categories (space-separated quoted strings)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Events to retrieve per category (default: 3)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"HF model override (default: {HF_MODEL})",
    )
    parser.add_argument(
        "--show-events",
        action="store_true",
        help="Print the raw retrieved events alongside the digest",
    )

    args = parser.parse_args()

    print("\n🌍 Generating World News Digest…")
    print("=" * 60)

    try:
        result = get_trends_digest(
            categories=args.categories,
            top_k=args.top_k,
            model=args.model,
        )
    except RuntimeError as e:
        print(f"\nError: {e}")
        sys.exit(1)

    if args.show_events:
        print(f"\n--- Retrieved Events ({result['total_events']}) ---")
        for category, events in result["categories"].items():
            if events:
                print(f"\n[{category.upper()}]")
                print(format_events_as_context(events))
        print("--- End Events ---\n")

    print(f"\nDigest (via {result['model']}):\n")
    print(result["digest"])
    print()
