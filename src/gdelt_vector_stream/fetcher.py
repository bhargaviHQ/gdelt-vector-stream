"""GDELT 2.0 Event data fetcher and parser."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_gdelt_events(csv_path: str | Path) -> list[dict[str, Any]]:
    """
    Load GDELT 2.0 Event CSV and extract relevant columns.

    Args:
        csv_path: Path to GDELT 2.0 Event CSV file

    Returns:
        List of event dictionaries with extracted fields
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    logger.info(f"Loading GDELT events from {csv_path}")

    # Required columns from GDELT 2.0 Event schema
    required_cols = [
        "GLOBALEVENTID",
        "SQLDATE",
        "EventCode",
        "EventBaseCode",
        "AvgTone",
        "NumMentions",
        "Actor1Name",
        "Actor2Name",
        "ActionGeo_CountryCode",
        "ActionGeo_Fullname",
        "ActionGeo_Lat",
        "ActionGeo_Long",
        "SourceURL",
    ]

    df = pd.read_csv(csv_path)

    # Check all required columns exist
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Select and rename for clarity
    df = df[required_cols].copy()

    # Convert to list of dicts
    events = df.to_dict(orient="records")
    logger.info(f"Loaded {len(events)} events")

    return events


def create_event_text(event: dict[str, Any]) -> str:
    """
    Create a human-readable text summary of an event for embedding.

    Args:
        event: Event dictionary from load_gdelt_events()

    Returns:
        Text summary suitable for embedding
    """
    # Handle NaN values from pandas
    def safe_get(d: dict, key: str, default: str) -> str:
        val = d.get(key, default)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        return str(val).strip() or default

    actor1 = safe_get(event, "Actor1Name", "Unknown")
    actor2 = safe_get(event, "Actor2Name", "Unknown")
    location = safe_get(event, "ActionGeo_Fullname", "Unknown location")
    tone = event.get("AvgTone", 0)
    if isinstance(tone, float) and np.isnan(tone):
        tone = 0

    # Map event codes to human-readable descriptions (simplified CAMEO codes)
    event_code = str(event.get("EventCode", ""))
    event_descriptions = {
        "06": "Military conflict",
        "17": "Protest",
        "19": "Use of force",
        "23": "Diplomatic cooperation",
        "24": "Appeal or request",
        "09": "Killing",
    }
    event_desc = event_descriptions.get(event_code[:2], f"Event type {event_code}")

    # Tone interpretation
    if tone > 50:
        tone_desc = "very positive"
    elif tone > 0:
        tone_desc = "positive"
    elif tone < -50:
        tone_desc = "very negative"
    elif tone < 0:
        tone_desc = "negative"
    else:
        tone_desc = "neutral"

    text = (
        f"{actor1} and {actor2} in {location}. "
        f"Event: {event_desc}. "
        f"Tone: {tone_desc} ({tone:.1f}). "
        f"Mentions: {event.get('NumMentions', 0)}"
    )

    return text
