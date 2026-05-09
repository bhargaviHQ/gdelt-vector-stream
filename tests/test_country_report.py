"""Unit tests for the Country Intelligence Report module."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gdelt_vector_stream.country_report import (
    REPORT_ANGLES,
    _gather_country_events,
    build_report_messages,
    compute_event_stats,
    get_country_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(vector_id: str, country: str = "UA", tone: float = -3.0, score: float = 0.88) -> dict:
    return {
        "vector_id": vector_id,
        "similarity_score": score,
        "metadata": {
            "actor1_name": "Government",
            "actor2_name": "Opposition",
            "country_code": country,
            "event_code": "17",
            "avg_tone": tone,
            "event_date": "20260101",
            "num_mentions": 10,
            "source_url": f"https://example.com/{vector_id}",
        },
    }


# ---------------------------------------------------------------------------
# compute_event_stats
# ---------------------------------------------------------------------------

def test_compute_event_stats_empty():
    stats = compute_event_stats([])
    assert stats["event_count"] == 0
    assert stats["avg_tone"] == 0.0
    assert stats["total_mentions"] == 0
    assert stats["top_actors"] == []
    assert stats["date_range"] is None
    print("✓ compute_event_stats: handles empty list")


def test_compute_event_stats_single_event():
    events = [_make_event("e1", tone=-5.0)]
    events[0]["metadata"]["num_mentions"] = 20
    stats = compute_event_stats(events)
    assert stats["event_count"] == 1
    assert stats["avg_tone"] == -5.0
    assert stats["total_mentions"] == 20
    assert "Government" in stats["top_actors"] or "Opposition" in stats["top_actors"]
    assert stats["date_range"] == {"earliest": "20260101", "latest": "20260101"}
    print("✓ compute_event_stats: correct stats for single event")


def test_compute_event_stats_multiple_events():
    events = [
        _make_event("e1", tone=2.0),
        _make_event("e2", tone=-4.0),
        _make_event("e3", tone=0.0),
    ]
    stats = compute_event_stats(events)
    assert stats["event_count"] == 3
    # avg of (2.0 + -4.0 + 0.0) / 3 = -0.667 ≈ -0.67
    assert abs(stats["avg_tone"] - (-2.0 / 3)) < 0.01
    assert stats["total_mentions"] == 30  # 10 * 3
    print("✓ compute_event_stats: averages tone and sums mentions across events")


def test_compute_event_stats_top_actors_sorted():
    """Actor appearing most frequently should rank first."""
    events = [
        _make_event("e1"),   # actor1=Government, actor2=Opposition
        _make_event("e2"),   # same actors
        _make_event("e3"),   # same actors
    ]
    stats = compute_event_stats(events)
    # Government and Opposition each appear 3 times; both should be present
    assert "Government" in stats["top_actors"]
    assert "Opposition" in stats["top_actors"]
    assert len(stats["top_actors"]) <= 5
    print("✓ compute_event_stats: top actors capped at 5 and sorted by frequency")


def test_compute_event_stats_date_range():
    events = [
        _make_event("e1"),
        _make_event("e2"),
    ]
    events[0]["metadata"]["event_date"] = "20260101"
    events[1]["metadata"]["event_date"] = "20260315"
    stats = compute_event_stats(events)
    assert stats["date_range"]["earliest"] == "20260101"
    assert stats["date_range"]["latest"] == "20260315"
    print("✓ compute_event_stats: date_range reflects earliest and latest dates")


# ---------------------------------------------------------------------------
# build_report_messages
# ---------------------------------------------------------------------------

def test_build_report_messages_structure():
    stats = {
        "event_count": 5,
        "avg_tone": -2.5,
        "total_mentions": 50,
        "top_actors": ["ActorA", "ActorB"],
        "date_range": None,
    }
    messages = build_report_messages("Ukraine", "Some event context here.", stats)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    # Country name should appear in both messages
    assert "Ukraine" in messages[0]["content"]
    assert "Ukraine" in messages[1]["content"]
    print("✓ build_report_messages: returns system + user message with country name")


def test_build_report_messages_tone_label():
    """Positive tone should use 'positive' label, negative should use 'negative'."""
    stats_pos = {"event_count": 1, "avg_tone": 3.5, "total_mentions": 5, "top_actors": [], "date_range": None}
    stats_neg = {"event_count": 1, "avg_tone": -3.5, "total_mentions": 5, "top_actors": [], "date_range": None}

    msgs_pos = build_report_messages("Germany", "ctx", stats_pos)
    msgs_neg = build_report_messages("Germany", "ctx", stats_neg)

    assert "positive" in msgs_pos[0]["content"]
    assert "negative" in msgs_neg[0]["content"]
    print("✓ build_report_messages: tone label reflects sign of avg_tone")


# ---------------------------------------------------------------------------
# _gather_country_events
# ---------------------------------------------------------------------------

def test_gather_country_events_queries_all_angles():
    """One search call per REPORT_ANGLE should be issued."""
    with patch("gdelt_vector_stream.country_report.semantic_search", return_value=[]) as mock_search:
        _gather_country_events("Brazil", top_k=3)

    assert mock_search.call_count == len(REPORT_ANGLES)
    # Each query should include the country name
    for c in mock_search.call_args_list:
        query = c.args[0]
        assert "Brazil" in query, f"Country name missing in query: {query!r}"
    print("✓ _gather_country_events: queries every REPORT_ANGLE with country name")


def test_gather_country_events_deduplicates():
    """An event that appears in multiple angle results should only be included once."""
    shared = _make_event("shared-001")
    unique_a = _make_event("unique-a")
    unique_b = _make_event("unique-b")

    call_num = 0

    def fake_search(query_text, top_k=5):
        nonlocal call_num
        call_num += 1
        if call_num == 1:
            return [shared, unique_a]
        if call_num == 2:
            return [shared, unique_b]
        return []

    with patch("gdelt_vector_stream.country_report.semantic_search", side_effect=fake_search):
        events = _gather_country_events("Iran", top_k=3)

    ids = [e["vector_id"] for e in events]
    assert ids.count("shared-001") == 1, "shared event should appear exactly once"
    assert "unique-a" in ids
    assert "unique-b" in ids
    print("✓ _gather_country_events: deduplicates events across angle queries")


# ---------------------------------------------------------------------------
# get_country_report
# ---------------------------------------------------------------------------

def test_get_country_report_full_flow():
    """End-to-end: mocked search + LLM should return expected structure."""
    mock_event = _make_event("full-001")

    with (
        patch("gdelt_vector_stream.country_report.semantic_search", return_value=[mock_event]),
        patch("gdelt_vector_stream.country_report.call_hf_inference", return_value="Test report text."),
    ):
        result = get_country_report(country="France", top_k=2)

    assert result["country"] == "France"
    assert result["report"] == "Test report text."
    assert len(result["events"]) > 0
    assert result["stats"]["event_count"] > 0
    assert result["model"] is not None
    print("✓ get_country_report: returns correct structure on full flow")


def test_get_country_report_no_events():
    """When no events are found, LLM should not be called and a helpful message returned."""
    with (
        patch("gdelt_vector_stream.country_report.semantic_search", return_value=[]),
        patch("gdelt_vector_stream.country_report.call_hf_inference") as mock_llm,
    ):
        result = get_country_report(country="XYZ", top_k=3)

    mock_llm.assert_not_called()
    assert result["country"] == "XYZ"
    assert "No events found" in result["report"]
    assert result["events"] == []
    assert result["stats"]["event_count"] == 0
    print("✓ get_country_report: skips LLM and returns helpful message when no events found")


def test_get_country_report_model_override():
    """A custom model should be forwarded to call_hf_inference."""
    mock_event = _make_event("model-001")
    custom_model = "Qwen/Qwen2.5-7B-Instruct"

    with (
        patch("gdelt_vector_stream.country_report.semantic_search", return_value=[mock_event]),
        patch("gdelt_vector_stream.country_report.call_hf_inference", return_value="ok") as mock_llm,
    ):
        result = get_country_report(country="Japan", top_k=1, model=custom_model)

    _, kwargs = mock_llm.call_args
    assert kwargs.get("model") == custom_model
    assert result["model"] == custom_model
    print("✓ get_country_report: model override forwarded to call_hf_inference")


def test_get_country_report_stats_propagated():
    """Stats computed from events should be included in the returned dict."""
    events = [_make_event(f"e{i}", tone=float(i)) for i in range(3)]

    with (
        patch("gdelt_vector_stream.country_report.semantic_search", side_effect=[events, [], [], [], []]),
        patch("gdelt_vector_stream.country_report.call_hf_inference", return_value="Report"),
    ):
        result = get_country_report(country="Kenya", top_k=3)

    stats = result["stats"]
    assert stats["event_count"] == 3
    assert "avg_tone" in stats
    assert "total_mentions" in stats
    print("✓ get_country_report: stats dict propagated correctly")


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running country_report tests...\n")
    try:
        test_compute_event_stats_empty()
        test_compute_event_stats_single_event()
        test_compute_event_stats_multiple_events()
        test_compute_event_stats_top_actors_sorted()
        test_compute_event_stats_date_range()
        test_build_report_messages_structure()
        test_build_report_messages_tone_label()
        test_gather_country_events_queries_all_angles()
        test_gather_country_events_deduplicates()
        test_get_country_report_full_flow()
        test_get_country_report_no_events()
        test_get_country_report_model_override()
        test_get_country_report_stats_propagated()
        print("\n✓ All country_report tests passed!")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
