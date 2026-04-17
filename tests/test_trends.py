"""Unit tests for the Trending Topics Digest module."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gdelt_vector_stream.trends import (
    DEFAULT_CATEGORIES,
    build_trends_prompt,
    get_trending_events,
    get_trends_digest,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(vector_id: str, country: str = "US", score: float = 0.9) -> dict:
    return {
        "vector_id": vector_id,
        "similarity_score": score,
        "metadata": {
            "actor1_name": "ActorA",
            "actor2_name": "ActorB",
            "country_code": country,
            "event_code": "17",
            "avg_tone": -2.5,
            "event_date": "20260101",
            "num_mentions": 5,
            "source_url": f"https://example.com/{vector_id}",
        },
    }


# ---------------------------------------------------------------------------
# build_trends_prompt
# ---------------------------------------------------------------------------

def test_build_trends_prompt_includes_categories():
    """Prompt should mention every category that has events."""
    category_events = {
        "military conflict and armed violence": [_make_event("evt-1")],
        "diplomacy and international cooperation": [_make_event("evt-2")],
        "protests and civil unrest": [],  # empty — should be skipped
    }
    prompt = build_trends_prompt(category_events)

    assert "Military conflict and armed violence" in prompt
    assert "Diplomacy and international cooperation" in prompt
    # Empty category should not appear
    assert "Protests and civil unrest" not in prompt
    print("✓ build_trends_prompt: includes non-empty categories, skips empty ones")


def test_build_trends_prompt_contains_event_data():
    """Prompt should include actor and source data from events."""
    category_events = {
        "economic policy and financial markets": [_make_event("eco-1", country="GB", score=0.85)],
    }
    prompt = build_trends_prompt(category_events)

    assert "ActorA" in prompt
    assert "GB" in prompt
    assert "https://example.com/eco-1" in prompt
    print("✓ build_trends_prompt: event metadata embedded in prompt")


def test_build_trends_prompt_all_empty():
    """Prompt with all empty categories should still be a valid string."""
    category_events = {"conflict": [], "diplomacy": []}
    prompt = build_trends_prompt(category_events)
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    print("✓ build_trends_prompt: handles all-empty categories gracefully")


# ---------------------------------------------------------------------------
# get_trending_events
# ---------------------------------------------------------------------------

def test_get_trending_events_deduplicates():
    """The same vector_id must not appear in more than one category."""
    shared_event = _make_event("shared-001")
    conflict_event = _make_event("conflict-only")
    diplomacy_event = _make_event("diplomacy-only")

    # 'conflict' returns shared + its own; 'diplomacy' returns shared + its own
    def fake_search(query_text, top_k=3):
        if "conflict" in query_text:
            return [shared_event, conflict_event]
        return [shared_event, diplomacy_event]

    with patch("gdelt_vector_stream.trends.semantic_search", side_effect=fake_search):
        result = get_trending_events(["conflict", "diplomacy"], top_k=2)

    conflict_ids = {e["vector_id"] for e in result["conflict"]}
    diplomacy_ids = {e["vector_id"] for e in result["diplomacy"]}

    # No overlap
    assert conflict_ids.isdisjoint(diplomacy_ids), "Duplicate vector_id across categories"
    # shared event claimed by the first category
    assert "shared-001" in conflict_ids
    assert "shared-001" not in diplomacy_ids
    print("✓ get_trending_events: deduplicates across categories")


def test_get_trending_events_empty_index():
    """When search returns nothing every category should be an empty list."""
    with patch("gdelt_vector_stream.trends.semantic_search", return_value=[]):
        result = get_trending_events(DEFAULT_CATEGORIES, top_k=3)

    assert all(len(v) == 0 for v in result.values())
    assert set(result.keys()) == set(DEFAULT_CATEGORIES)
    print("✓ get_trending_events: handles empty index")


def test_get_trending_events_respects_top_k():
    """top_k should be forwarded to semantic_search on every category call."""
    with patch("gdelt_vector_stream.trends.semantic_search", return_value=[]) as mock_search:
        get_trending_events(["conflict", "diplomacy"], top_k=3)

    for call in mock_search.call_args_list:
        assert call.kwargs.get("top_k") == 3 or call.args[1] == 3
    print("✓ get_trending_events: top_k forwarded to semantic_search")


# ---------------------------------------------------------------------------
# get_trends_digest
# ---------------------------------------------------------------------------

def test_get_trends_digest_full_flow():
    """End-to-end: mocked search + LLM should return expected structure."""
    mock_event = _make_event("full-001")

    with (
        patch("gdelt_vector_stream.trends.semantic_search", return_value=[mock_event]),
        patch("gdelt_vector_stream.trends.call_hf_inference", return_value="Global digest text."),
    ):
        result = get_trends_digest(categories=["conflict", "diplomacy"], top_k=2)

    assert result["digest"] == "Global digest text."
    assert "conflict" in result["categories"]
    assert "diplomacy" in result["categories"]
    assert result["total_events"] > 0
    assert result["model"] is not None
    print("✓ get_trends_digest: correct structure on full flow")


def test_get_trends_digest_no_events():
    """Digest with zero events should return a helpful message without calling LLM."""
    with (
        patch("gdelt_vector_stream.trends.semantic_search", return_value=[]),
        patch("gdelt_vector_stream.trends.call_hf_inference") as mock_llm,
    ):
        result = get_trends_digest(categories=["conflict"], top_k=3)

    mock_llm.assert_not_called()
    assert "No events found" in result["digest"]
    assert result["total_events"] == 0
    print("✓ get_trends_digest: skips LLM when no events found")


def test_get_trends_digest_uses_default_categories():
    """When categories=None, DEFAULT_CATEGORIES should be used."""
    with (
        patch("gdelt_vector_stream.trends.semantic_search", return_value=[]) as mock_search,
        patch("gdelt_vector_stream.trends.call_hf_inference", return_value="ok"),
    ):
        get_trends_digest()

    called_queries = [call.args[0] for call in mock_search.call_args_list]
    for cat in DEFAULT_CATEGORIES:
        assert cat in called_queries, f"Expected {cat!r} to be searched"
    print("✓ get_trends_digest: uses DEFAULT_CATEGORIES when none specified")


def test_get_trends_digest_model_override():
    """Model override should be passed through to call_hf_inference."""
    mock_event = _make_event("model-001")
    custom_model = "mistralai/Mistral-7B-Instruct-v0.3"

    with (
        patch("gdelt_vector_stream.trends.semantic_search", return_value=[mock_event]),
        patch("gdelt_vector_stream.trends.call_hf_inference", return_value="ok") as mock_llm,
    ):
        result = get_trends_digest(categories=["conflict"], top_k=1, model=custom_model)

    _, kwargs = mock_llm.call_args
    assert kwargs.get("model") == custom_model
    assert result["model"] == custom_model
    print("✓ get_trends_digest: model override forwarded to LLM")


# ---------------------------------------------------------------------------
# DEFAULT_CATEGORIES sanity check
# ---------------------------------------------------------------------------

def test_default_categories_count():
    """There should be 8 default categories."""
    assert len(DEFAULT_CATEGORIES) == 8, f"Expected 8, got {len(DEFAULT_CATEGORIES)}"
    print(f"✓ DEFAULT_CATEGORIES has {len(DEFAULT_CATEGORIES)} entries")


def test_default_categories_are_strings():
    assert all(isinstance(c, str) and c for c in DEFAULT_CATEGORIES)
    print("✓ DEFAULT_CATEGORIES are all non-empty strings")


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running trends tests...\n")
    try:
        test_build_trends_prompt_includes_categories()
        test_build_trends_prompt_contains_event_data()
        test_build_trends_prompt_all_empty()
        test_get_trending_events_deduplicates()
        test_get_trending_events_empty_index()
        test_get_trending_events_respects_top_k()
        test_get_trends_digest_full_flow()
        test_get_trends_digest_no_events()
        test_get_trends_digest_uses_default_categories()
        test_get_trends_digest_model_override()
        test_default_categories_count()
        test_default_categories_are_strings()
        print("\n✓ All trends tests passed!")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
