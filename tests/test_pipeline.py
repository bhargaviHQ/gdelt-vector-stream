"""Basic tests for the ingestion pipeline."""

import sys
from pathlib import Path

# Add src to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gdelt_vector_stream.embedder import create_pinecone_vectors, embed_event, get_embedder
from gdelt_vector_stream.fetcher import create_event_text, load_gdelt_events


def test_load_events():
    """Test loading sample GDELT CSV."""
    csv_path = Path(__file__).parent.parent / "data" / "sample_gdelt_events.csv"
    events = load_gdelt_events(csv_path)

    assert len(events) > 0, "Should load events"
    assert "GLOBALEVENTID" in events[0], "Event should have GLOBALEVENTID"
    print(f"✓ Loaded {len(events)} events")


def test_event_text():
    """Test event text generation."""
    csv_path = Path(__file__).parent.parent / "data" / "sample_gdelt_events.csv"
    events = load_gdelt_events(csv_path)

    text = create_event_text(events[0])
    assert isinstance(text, str), "Event text should be a string"
    assert len(text) > 0, "Event text should not be empty"
    print(f"✓ Generated event text: {text[:80]}...")


def test_embedding():
    """Test embedding generation."""
    embedder = get_embedder()

    text = "United States Government and Protest Group demonstrated in New York"
    embedding = embed_event(text, embedder)

    assert isinstance(embedding, list), "Embedding should be a list"
    assert len(embedding) == 384, "Sentence-Transformers all-MiniLM should produce 384-dim vectors"
    assert all(isinstance(x, float) for x in embedding), "Embedding values should be floats"
    print("✓ Generated 384-dim embedding")


def test_pinecone_vectors():
    """Test Pinecone vector creation."""
    csv_path = Path(__file__).parent.parent / "data" / "sample_gdelt_events.csv"
    events = load_gdelt_events(csv_path)

    embedder = get_embedder()
    event_texts = [create_event_text(e) for e in events]
    embeddings = [embed_event(t, embedder) for t in event_texts]

    vectors = create_pinecone_vectors(events, event_texts, embeddings)

    assert len(vectors) == len(events), "Should create vector for each event"

    vec_id, embedding, metadata = vectors[0]
    assert vec_id.startswith("gdelt-"), "Vector ID should start with 'gdelt-'"
    assert len(embedding) == 384, "Vector should be 384-dim"
    assert "event_date" in metadata, "Metadata should have event_date"
    assert "country_code" in metadata, "Metadata should have country_code"

    print(f"✓ Created {len(vectors)} Pinecone vectors")
    print(f"  Sample vector ID: {vec_id}")
    print(f"  Metadata keys: {list(metadata.keys())}")


if __name__ == "__main__":
    print("Running pipeline tests...\n")

    try:
        test_load_events()
        test_event_text()
        test_embedding()
        test_pinecone_vectors()

        print("\n✓ All tests passed!")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
