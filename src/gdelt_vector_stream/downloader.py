"""GDELT 2.0 auto-downloader: fetch, sample, and ingest real GDELT data."""

import argparse
import csv
import hashlib
import io
import json
import logging
import os
import time
import zipfile
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from gdelt_vector_stream.embedder import create_pinecone_vectors, embed_events_batch, get_embedder
from gdelt_vector_stream.fetcher import create_event_text
from gdelt_vector_stream.ingestor import ingest_vectors

load_dotenv()

logger = logging.getLogger(__name__)

# --- Constants ---

GDELT_MANIFEST_URL = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
DEFAULT_SAMPLE_SIZE = 100
POLL_INTERVAL_SECONDS = 900  # 15 minutes
# Anchor to the project root (two levels above src/gdelt_vector_stream/) so the path
# is stable regardless of the working directory when the module is run.
PROCESSED_FILE_PATH = Path(__file__).resolve().parents[2] / "data" / ".processed_files.json"

# GDELT 2.0 Event export: 61 tab-delimited columns, NO header row.
# Column names in positional order per GDELT documentation.
GDELT_COLUMNS = [
    "GLOBALEVENTID", "SQLDATE", "MonthYear", "Year", "FractionDate",
    "Actor1Code", "Actor1Name", "Actor1CountryCode",
    "Actor1KnownGroupCode", "Actor1EthnicCode", "Actor1Religion1Code",
    "Actor1Religion2Code", "Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code",
    "Actor2Code", "Actor2Name", "Actor2CountryCode",
    "Actor2KnownGroupCode", "Actor2EthnicCode", "Actor2Religion1Code",
    "Actor2Religion2Code", "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code",
    "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode",
    "QuadClass", "GoldsteinScale", "NumMentions", "NumSources", "NumArticles",
    "AvgTone",
    "Actor1Geo_Type", "Actor1Geo_FullName", "Actor1Geo_CountryCode",
    "Actor1Geo_ADM1Code", "Actor1Geo_ADM2Code", "Actor1Geo_Lat", "Actor1Geo_Long",
    "Actor1Geo_FeatureID",
    "Actor2Geo_Type", "Actor2Geo_FullName", "Actor2Geo_CountryCode",
    "Actor2Geo_ADM1Code", "Actor2Geo_ADM2Code", "Actor2Geo_Lat", "Actor2Geo_Long",
    "Actor2Geo_FeatureID",
    "ActionGeo_Type", "ActionGeo_FullName", "ActionGeo_CountryCode",
    "ActionGeo_ADM1Code", "ActionGeo_ADM2Code", "ActionGeo_Lat", "ActionGeo_Long",
    "ActionGeo_FeatureID",
    "DATEADDED", "SOURCEURL",
]

# Columns that need renaming to match what fetcher.py / embedder.py expect
COLUMN_RENAME = {
    "ActionGeo_FullName": "ActionGeo_Fullname",
    "SOURCEURL": "SourceURL",
}

# Columns the pipeline actually needs (from fetcher.py required_cols)
REQUIRED_COLUMNS = [
    "GLOBALEVENTID", "SQLDATE", "EventCode", "EventBaseCode",
    "AvgTone", "NumMentions", "Actor1Name", "Actor2Name",
    "ActionGeo_CountryCode", "ActionGeo_Fullname",
    "ActionGeo_Lat", "ActionGeo_Long", "SourceURL",
]

# Numeric fields that need type conversion
NUMERIC_FLOAT = {"AvgTone", "ActionGeo_Lat", "ActionGeo_Long"}
NUMERIC_INT = {"GLOBALEVENTID", "SQLDATE", "NumMentions"}


# --- Manifest & URL Parsing ---

def fetch_manifest(url: str = GDELT_MANIFEST_URL) -> str:
    """Download the GDELT master manifest file."""
    logger.info(f"Fetching GDELT manifest from {url}...")
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    logger.info(f"Manifest downloaded ({len(response.text) // 1024} KB)")
    return response.text


def parse_export_urls(manifest_text: str) -> list[str]:
    """
    Parse manifest and extract .export.CSV.zip URLs, newest first.

    Each line: <size> <md5> <url>
    """
    urls = []
    for line in manifest_text.strip().splitlines():
        parts = line.strip().split()
        if len(parts) >= 3:
            url = parts[-1]
            if url.endswith(".export.CSV.zip"):
                urls.append(url)

    # Newest files are at the end of the manifest
    urls.reverse()
    return urls


def parse_manifest_md5s(manifest_text: str) -> dict[str, str]:
    """
    Build a mapping of URL → expected MD5 from the manifest.

    Each line: <size> <md5> <url>
    """
    md5s: dict[str, str] = {}
    for line in manifest_text.strip().splitlines():
        parts = line.strip().split()
        if len(parts) >= 3:
            md5s[parts[-1]] = parts[1]
    return md5s


# --- ZIP Download & Sampling ---

def download_and_sample(
    zip_url: str,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    expected_md5: str | None = None,
) -> list[dict[str, Any]]:
    """
    Download a GDELT .export.CSV.zip and parse first N records in-memory.

    If *expected_md5* is provided the downloaded content is verified against it
    (MD5 hashes are published in the GDELT master manifest). A mismatch raises
    ValueError to prevent ingesting corrupt or tampered data.

    Returns list of event dicts matching the format from fetcher.load_gdelt_events().
    """
    logger.info(f"Downloading {zip_url}...")
    response = requests.get(zip_url, timeout=60)
    response.raise_for_status()

    # Verify MD5 integrity when the manifest-supplied hash is available
    if expected_md5:
        actual_md5 = hashlib.md5(response.content).hexdigest()
        if actual_md5 != expected_md5:
            raise ValueError(
                f"MD5 mismatch for {zip_url}: expected {expected_md5}, got {actual_md5}"
            )
        logger.debug(f"MD5 verified: {actual_md5}")

    # Open ZIP in-memory
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        csv_name = zf.namelist()[0]
        logger.info(f"Parsing {csv_name} (sampling {sample_size} records)...")

        with zf.open(csv_name) as f:
            reader = csv.reader(
                io.TextIOWrapper(f, encoding="utf-8", errors="replace"),
                delimiter="\t",
            )

            events = []
            skipped = 0

            for row_num, row in enumerate(reader):
                if len(events) >= sample_size:
                    break

                # Pad short rows with empty strings
                if len(row) < len(GDELT_COLUMNS):
                    row.extend([""] * (len(GDELT_COLUMNS) - len(row)))

                # Skip rows with too many columns (malformed)
                if len(row) > len(GDELT_COLUMNS) + 5:
                    skipped += 1
                    continue

                # Map positional values to column names
                raw = dict(zip(GDELT_COLUMNS, row[: len(GDELT_COLUMNS)]))

                # Rename columns to match pipeline expectations
                for old_name, new_name in COLUMN_RENAME.items():
                    if old_name in raw:
                        raw[new_name] = raw.pop(old_name)

                # Convert numeric fields
                event = {}
                for col in REQUIRED_COLUMNS:
                    val = raw.get(col, "")
                    if col in NUMERIC_FLOAT:
                        try:
                            event[col] = float(val) if val else 0.0
                        except ValueError:
                            event[col] = 0.0
                    elif col in NUMERIC_INT:
                        try:
                            event[col] = int(val) if val else 0
                        except ValueError:
                            event[col] = 0
                    else:
                        event[col] = val

                events.append(event)

            if skipped:
                logger.warning(f"Skipped {skipped} malformed rows")

    logger.info(f"Parsed {len(events)} events from {csv_name}")
    return events


# --- Processed File Tracking ---

def load_processed(path: Path = PROCESSED_FILE_PATH) -> set[str]:
    """Load set of already-processed file URLs."""
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text())
        return set(data.get("processed", []))
    except (json.JSONDecodeError, KeyError):
        logger.warning(f"Corrupted processed file at {path}, starting fresh")
        return set()


def save_processed(url: str, path: Path = PROCESSED_FILE_PATH) -> None:
    """Add a URL to the processed files tracker."""
    processed = load_processed(path)
    processed.add(url)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"processed": sorted(processed)}, indent=2))


# --- Pipeline Integration ---

def run_pipeline(events: list[dict[str, Any]], dry_run: bool = False) -> dict[str, Any]:
    """
    Run embed → ingest pipeline on parsed events.

    Reuses existing pipeline functions directly.
    """
    if not events:
        logger.warning("No events to process")
        return {"total_vectors": 0, "success": False}

    # Create text summaries
    event_texts = [create_event_text(event) for event in events]

    if dry_run:
        logger.info(f"[DRY RUN] Would embed and ingest {len(events)} events")
        logger.info(f"[DRY RUN] Sample text: {event_texts[0][:100]}...")
        return {"total_vectors": len(events), "total_upserted": 0, "success": True, "dry_run": True}

    # Embed
    embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embedder = get_embedder(embedding_model)
    embeddings = embed_events_batch(event_texts, embedder)

    # Create Pinecone vectors and ingest
    vectors = create_pinecone_vectors(events, event_texts, embeddings)
    return ingest_vectors(vectors)


# --- Orchestration ---

def download_and_ingest(
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    max_files: int = 1,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """
    Main orchestration: fetch manifest → find new files → download → sample → ingest.

    Returns list of ingestion summaries (one per file processed).
    """
    processed = load_processed()
    manifest = fetch_manifest()
    all_urls = parse_export_urls(manifest)
    md5_index = parse_manifest_md5s(manifest)

    new_urls = [u for u in all_urls if u not in processed][:max_files]

    if not new_urls:
        logger.info("No new GDELT files to process")
        return []

    logger.info(f"Found {len(new_urls)} new file(s) to process")

    summaries = []
    for url in new_urls:
        try:
            events = download_and_sample(url, sample_size, expected_md5=md5_index.get(url))
            summary = run_pipeline(events, dry_run=dry_run)
            summary["source_url"] = url
            summary["events_sampled"] = len(events)
            summaries.append(summary)

            # Only mark as processed if ingestion succeeded
            if summary.get("success"):
                save_processed(url)
                logger.info(f"✓ Processed: {url}")
            else:
                logger.error(f"✗ Ingestion failed for {url}, will retry next run")

        except Exception as e:
            logger.error(f"✗ Failed to process {url}: {e}")
            continue

    return summaries


def watch(
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    poll_interval: int = POLL_INTERVAL_SECONDS,
    dry_run: bool = False,
) -> None:
    """Continuously poll for new GDELT files and ingest them."""
    logger.info(f"Watch mode: polling every {poll_interval}s for new GDELT files")
    logger.info(f"Sample size: {sample_size} records per file")
    logger.info("Press Ctrl+C to stop\n")

    try:
        while True:
            summaries = download_and_ingest(
                sample_size=sample_size, max_files=1, dry_run=dry_run
            )

            if not summaries:
                logger.info(f"No new files. Sleeping {poll_interval}s...")

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        logger.info("\nWatch mode stopped by user.")


# --- CLI ---

if __name__ == "__main__":
    log_level = os.getenv("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="GDELT 2.0 auto-downloader: fetch, sample, and ingest real GDELT data"
    )
    parser.add_argument(
        "--sample", type=int, default=DEFAULT_SAMPLE_SIZE,
        help=f"Number of records to sample per file (default: {DEFAULT_SAMPLE_SIZE})",
    )
    parser.add_argument(
        "--max-files", type=int, default=1,
        help="Max files to process per run (default: 1)",
    )
    parser.add_argument(
        "--watch", action="store_true",
        help="Enable continuous polling mode (every 15 min)",
    )
    parser.add_argument(
        "--poll-interval", type=int, default=POLL_INTERVAL_SECONDS,
        help=f"Seconds between polls in watch mode (default: {POLL_INTERVAL_SECONDS})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Fetch and parse but don't ingest (for testing)",
    )

    args = parser.parse_args()

    if args.watch:
        watch(
            sample_size=args.sample,
            poll_interval=args.poll_interval,
            dry_run=args.dry_run,
        )
    else:
        summaries = download_and_ingest(
            sample_size=args.sample,
            max_files=args.max_files,
            dry_run=args.dry_run,
        )

        if summaries and all(s.get("success") for s in summaries):
            logger.info("✓ All files processed successfully!")
        elif not summaries:
            logger.info("No new files to process.")
        else:
            logger.error("✗ Some files failed. Check logs above.")
