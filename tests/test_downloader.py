"""Tests for the GDELT downloader module."""

import tempfile
from pathlib import Path

from gdelt_vector_stream.downloader import (
    GDELT_COLUMNS,
    REQUIRED_COLUMNS,
    load_processed,
    parse_export_urls,
    save_processed,
)


# --- Sample manifest data ---
SAMPLE_MANIFEST = """150383 297a16b493de7cf6ca809a7cc31d0b93 http://data.gdeltproject.org/gdeltv2/20260405230000.export.CSV.zip
318084 bb27f78ba45f69a17ea6ed7755e9f8ff http://data.gdeltproject.org/gdeltv2/20260405230000.mentions.CSV.zip
10768507 ea8dde0beb0ba98810a92db068c0ce99 http://data.gdeltproject.org/gdeltv2/20260405230000.gkg.csv.zip
149211 2a91041d7e72b0fc6a629e2ff867b240 http://data.gdeltproject.org/gdeltv2/20260405231500.export.CSV.zip
339037 dec3f427076b716a8112b9086c342523 http://data.gdeltproject.org/gdeltv2/20260405231500.mentions.CSV.zip
"""


def test_parse_export_urls():
    """Only .export.CSV.zip URLs should be extracted, newest first."""
    urls = parse_export_urls(SAMPLE_MANIFEST)

    assert len(urls) == 2, f"Expected 2 export URLs, got {len(urls)}"
    # Newest first (reversed order)
    assert "20260405231500" in urls[0]
    assert "20260405230000" in urls[1]
    # No mentions or gkg
    assert all(".export.CSV.zip" in u for u in urls)
    print("✓ parse_export_urls: filters correctly, newest first")


def test_parse_export_urls_empty():
    """Empty manifest should return empty list."""
    assert parse_export_urls("") == []
    assert parse_export_urls("   \n  \n  ") == []
    print("✓ parse_export_urls: handles empty manifest")


def test_processed_file_tracking():
    """Processed file tracker should persist and deduplicate."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / ".processed_files.json"

        # Initially empty
        assert load_processed(path) == set()

        # Save one
        save_processed("http://example.com/file1.zip", path)
        loaded = load_processed(path)
        assert "http://example.com/file1.zip" in loaded
        assert len(loaded) == 1

        # Save another
        save_processed("http://example.com/file2.zip", path)
        loaded = load_processed(path)
        assert len(loaded) == 2

        # Duplicate should not increase count
        save_processed("http://example.com/file1.zip", path)
        loaded = load_processed(path)
        assert len(loaded) == 2

    print("✓ processed file tracking: persists and deduplicates")


def test_processed_file_corrupted():
    """Corrupted JSON should return empty set, not crash."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / ".processed_files.json"
        path.write_text("not valid json {{{")

        result = load_processed(path)
        assert result == set()

    print("✓ processed file tracking: handles corrupted JSON")


def test_column_count():
    """GDELT_COLUMNS should have exactly 61 entries."""
    assert len(GDELT_COLUMNS) == 61, f"Expected 61 columns, got {len(GDELT_COLUMNS)}"
    print("✓ GDELT_COLUMNS has 61 entries")


def test_required_columns_subset():
    """All required columns should map to GDELT columns (after rename)."""
    # After rename: ActionGeo_FullName → ActionGeo_Fullname, SOURCEURL → SourceURL
    gdelt_renamed = set(GDELT_COLUMNS)
    gdelt_renamed.discard("ActionGeo_FullName")
    gdelt_renamed.add("ActionGeo_Fullname")
    gdelt_renamed.discard("SOURCEURL")
    gdelt_renamed.add("SourceURL")

    for col in REQUIRED_COLUMNS:
        assert col in gdelt_renamed, f"Required column {col} not found in GDELT columns"

    print("✓ All required columns map to GDELT columns")


if __name__ == "__main__":
    print("Running downloader tests...\n")

    try:
        test_parse_export_urls()
        test_parse_export_urls_empty()
        test_processed_file_tracking()
        test_processed_file_corrupted()
        test_column_count()
        test_required_columns_subset()
        print("\n✓ All downloader tests passed!")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
