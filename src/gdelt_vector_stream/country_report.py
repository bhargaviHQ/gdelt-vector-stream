"""Country Intelligence Report: generate a focused briefing on GDELT events for a specific country."""

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

# Angle suffixes used to retrieve events from multiple perspectives for the target country
REPORT_ANGLES = [
    "military conflict and violence",
    "diplomacy and international relations",
    "economic policy and trade",
    "protests and civil unrest",
    "humanitarian situation",
]


def _gather_country_events(
    country: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Retrieve GDELT events related to a country from multiple search angles.

    Runs one semantic search per angle query (e.g. "Ukraine military conflict")
    and deduplicates results so each vector appears at most once.

    Args:
        country: Country name or code (e.g. "Ukraine", "BR").
        top_k: Number of results to retrieve per angle query.

    Returns:
        Deduplicated list of event dicts, ordered by angle then similarity.
    """
    seen_ids: set[str] = set()
    events: list[dict[str, Any]] = []

    for angle in REPORT_ANGLES:
        query = f"{country} {angle}"
        logger.info(f"Searching: {query!r}")
        results = semantic_search(query, top_k=top_k)

        for event in results:
            vid = event["vector_id"]
            if vid not in seen_ids:
                seen_ids.add(vid)
                events.append(event)

    logger.info(f"Gathered {len(events)} unique events for {country!r}")
    return events


def compute_event_stats(events: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute summary statistics over a list of GDELT events.

    Args:
        events: List of event dicts (each with a ``metadata`` sub-dict).

    Returns:
        Dict with ``avg_tone``, ``total_mentions``, ``top_actors``,
        ``event_count``, and ``date_range``.
    """
    if not events:
        return {
            "event_count": 0,
            "avg_tone": 0.0,
            "total_mentions": 0,
            "top_actors": [],
            "date_range": None,
        }

    tones: list[float] = []
    total_mentions = 0
    actor_counts: dict[str, int] = {}
    dates: list[str] = []

    for event in events:
        meta = event["metadata"]

        tone = meta.get("avg_tone", 0)
        try:
            tones.append(float(tone))
        except (TypeError, ValueError):
            pass

        mentions = meta.get("num_mentions", 0)
        try:
            total_mentions += int(mentions)
        except (TypeError, ValueError):
            pass

        for actor_key in ("actor1_name", "actor2_name"):
            name = meta.get(actor_key, "")
            if name and name != "Unknown":
                actor_counts[name] = actor_counts.get(name, 0) + 1

        date = meta.get("event_date", "")
        if date:
            dates.append(str(date))

    avg_tone = sum(tones) / len(tones) if tones else 0.0
    top_actors = sorted(actor_counts, key=lambda k: actor_counts[k], reverse=True)[:5]
    dates_sorted = sorted(dates)
    date_range = (
        {"earliest": dates_sorted[0], "latest": dates_sorted[-1]} if dates_sorted else None
    )

    return {
        "event_count": len(events),
        "avg_tone": round(avg_tone, 2),
        "total_mentions": total_mentions,
        "top_actors": top_actors,
        "date_range": date_range,
    }


def build_report_messages(country: str, context: str, stats: dict[str, Any]) -> list[dict[str, str]]:
    """
    Build the chat messages for a country intelligence report.

    Args:
        country: Country name used as the report subject.
        context: Formatted event context block.
        stats: Summary statistics from ``compute_event_stats``.

    Returns:
        List of role/content dicts ready for the HF Inference API.
    """
    tone_label = "positive" if stats["avg_tone"] > 0 else "negative"
    system_content = (
        f"You are a geopolitical intelligence analyst. Produce a structured Country Intelligence "
        f"Report for **{country}** using ONLY the GDELT news events provided below. "
        f"Structure your report with these sections:\n"
        f"1. Executive Summary (2-3 sentences)\n"
        f"2. Key Developments (bullet points covering the most significant events)\n"
        f"3. Tone Assessment (the overall sentiment is {tone_label} at {stats['avg_tone']:.1f}; "
        f"explain what this means for the situation)\n"
        f"4. Key Actors (mention the most active parties)\n"
        f"5. Sources\n\n"
        f"Index statistics: {stats['event_count']} events, "
        f"{stats['total_mentions']} total media mentions.\n\n"
        f"--- GDELT NEWS EVENTS ---\n{context}\n--- END EVENTS ---"
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": f"Generate the Country Intelligence Report for {country}."},
    ]


def get_country_report(
    country: str,
    top_k: int = 5,
    model: str | None = None,
) -> dict[str, Any]:
    """
    Generate a Country Intelligence Report grounded in GDELT events.

    Args:
        country: Country name or code to report on (e.g. ``"Ukraine"``, ``"BR"``).
        top_k: Events to retrieve per search angle (total events ≤ top_k × angles).
        model: HF model override; reads ``HF_MODEL`` env var if not given.

    Returns:
        Dict with keys:

        - ``report``: LLM-generated intelligence report string.
        - ``country``: The queried country.
        - ``events``: List of deduplicated event dicts used for the report.
        - ``stats``: Summary statistics (tone, mentions, actors, date range).
        - ``model``: Model ID used.
    """
    model = model or HF_MODEL

    # Step 1: retrieve events across multiple angles
    events = _gather_country_events(country, top_k=top_k)

    if not events:
        return {
            "report": (
                f"No events found for '{country}' in the GDELT index. "
                "Try ingesting more data or use a different country name."
            ),
            "country": country,
            "events": [],
            "stats": compute_event_stats([]),
            "model": model,
        }

    # Step 2: compute statistics
    stats = compute_event_stats(events)
    logger.info(
        f"Stats for {country!r}: {stats['event_count']} events, "
        f"avg tone {stats['avg_tone']:.2f}, {stats['total_mentions']} mentions"
    )

    # Step 3: format context and build messages
    context = format_events_as_context(events)
    messages = build_report_messages(country, context, stats)

    # Step 4: call LLM
    logger.info(f"Sending country report prompt to {model}...")
    report = call_hf_inference(messages, model=model)

    return {
        "report": report,
        "country": country,
        "events": events,
        "stats": stats,
        "model": model,
    }


# --- CLI ---

if __name__ == "__main__":
    log_level = os.getenv("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Generate a Country Intelligence Report from your GDELT vector index"
    )
    parser.add_argument("country", help="Country name or code (e.g. 'Ukraine', 'BR')")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Events to retrieve per search angle (default: 5)",
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
        help="Print the raw retrieved events alongside the report",
    )

    args = parser.parse_args()

    print(f"\n🗺️  Generating Country Intelligence Report for: {args.country}")
    print("=" * 60)

    try:
        result = get_country_report(
            country=args.country,
            top_k=args.top_k,
            model=args.model,
        )
    except RuntimeError as e:
        print(f"\nError: {e}")
        sys.exit(1)

    if args.show_events:
        print(f"\n--- Retrieved Events ({len(result['events'])}) ---")
        print(format_events_as_context(result["events"]))
        print("--- End Events ---\n")

    stats = result["stats"]
    print(f"\nStatistics:")
    print(f"  Events analyzed : {stats['event_count']}")
    print(f"  Average tone    : {stats['avg_tone']:.2f}")
    print(f"  Total mentions  : {stats['total_mentions']}")
    if stats["top_actors"]:
        print(f"  Top actors      : {', '.join(stats['top_actors'])}")
    if stats["date_range"]:
        dr = stats["date_range"]
        print(f"  Date range      : {dr['earliest']} – {dr['latest']}")

    print(f"\nReport (via {result['model']}):\n")
    print(result["report"])
    print()
