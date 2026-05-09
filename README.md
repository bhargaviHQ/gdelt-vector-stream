# gdelt-vector-stream

Ingest news events from the [GDELT Project](https://www.gdeltproject.org/), embed them as dense vectors, store them in [Pinecone](https://www.pinecone.io/), and query them with a RAG-powered analyst backed by Hugging Face Inference API.

## How it works

```
GDELT 2.0 Events (.csv)
        |
   [ fetcher.py ]       Download & parse GDELT event files
        |
   [ embedder.py ]      Generate 384-dim vectors (Sentence-Transformers, local)
        |
   [ ingestor.py ]      Batch-upsert 100 vectors at a time into Pinecone
        |
   [ query.py ]         Semantic similarity search over the index
        |
   [ analyst.py ]       RAG: retrieve events + answer questions via HF Inference API
```

## Quick start

```bash
# 1. Clone and set up
git clone https://github.com/bhargaviHQ/gdelt-vector-stream.git
cd gdelt-vector-stream
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"

# 2. Configure credentials
cp .env.example .env
# Fill in PINECONE_API_KEY, HF_TOKEN (free from https://huggingface.co/settings/tokens)
```

## Usage

### Ingest GDELT data

```bash
# Download and ingest the latest GDELT events (100 events per file, 1 file)
python -m gdelt_vector_stream.downloader

# Ingest more data for better search results
python -m gdelt_vector_stream.downloader --max-files 10 --sample 100

# Watch mode: poll every 15 minutes for new files
python -m gdelt_vector_stream.downloader --watch
```

### Semantic search

```bash
python -m gdelt_vector_stream.query "renewable energy developments worldwide"
```

### RAG analyst

Ask natural language questions grounded in your GDELT data:

```bash
python -m gdelt_vector_stream.analyst "What are the latest developments in global diplomacy?"

# Show retrieved events alongside the answer
python -m gdelt_vector_stream.analyst "Recent international space exploration news" --show-context

# Use a different model
python -m gdelt_vector_stream.analyst "Global trends in renewable energy adoption" --model Qwen/Qwen2.5-7B-Instruct
```

### Trending Topics Digest

```bash
# Generate a World News Digest across 8 default topic categories
python -m gdelt_vector_stream.trends

# Show the raw retrieved events alongside the digest
python -m gdelt_vector_stream.trends --show-events

# Customise which topics are covered
python -m gdelt_vector_stream.trends --categories "military conflict" "energy crisis" "elections"

# Retrieve more events per category for a richer digest
python -m gdelt_vector_stream.trends --top-k 5

# Use a different model
python -m gdelt_vector_stream.trends --model Qwen/Qwen2.5-7B-Instruct
```

The digest searches 8 broad categories (conflict, diplomacy, protests, economy, environment, health, humanitarian, technology), deduplicates events across them, and asks the LLM for one paragraph per active topic.

### Country Intelligence Report

Generate a focused, LLM-backed intelligence briefing for any country or region — no manual query construction needed:

```bash
# Generate a report for Ukraine
python -m gdelt_vector_stream.country_report Ukraine

# Show the raw retrieved events alongside the report
python -m gdelt_vector_stream.country_report France --show-events

# Retrieve more events per search angle for a richer report
python -m gdelt_vector_stream.country_report Brazil --top-k 8

# Use a different model
python -m gdelt_vector_stream.country_report Japan --model Qwen/Qwen2.5-7B-Instruct
```

The report searches five thematic angles (military conflict, diplomacy, economic policy, protests, humanitarian situation) for the specified country, deduplicates events across angles, computes tone/mention statistics, and asks the LLM to produce a structured briefing with an Executive Summary, Key Developments, Tone Assessment, Key Actors, and Sources sections.

### API server

A FastAPI backend is available for building frontends:

```bash
pip install -e ".[api]"
uvicorn api.server:app --reload
```

Endpoints:
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Check Pinecone + HF connectivity |
| GET | `/api/stats` | Index statistics (vector count, etc.) |
| GET | `/api/search?q=...&top_k=5` | Semantic search |
| POST | `/api/ask` | RAG analyst (JSON body: `question`, `top_k`, `model`) |
| POST | `/api/ingest` | Trigger ingestion (JSON body: `sample_size`, `max_files`) |
| GET | `/api/trends?top_k=3` | World News Digest across default topic categories |
| GET | `/api/country-report?country=Ukraine` | Country Intelligence Report for a specific country |

## Tech stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.11+ |
| Vector DB | Pinecone (free tier) |
| Data source | GDELT 2.0 Events |
| Embeddings | Sentence-Transformers `all-MiniLM-L6-v2` (384-dim, local) |
| LLM (RAG) | Hugging Face Inference API (`Meta-Llama-3-8B-Instruct`) |
| API | FastAPI |
| Testing | pytest |

## Environment variables

See [`.env.example`](.env.example) for the full list. Key variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `PINECONE_API_KEY` | Yes | Pinecone API key |
| `PINECONE_INDEX_NAME` | Yes | Index name (default: `gdelt-events`) |
| `HF_TOKEN` | For RAG | Free token from HuggingFace |
| `HF_MODEL` | No | HF model ID (default: `meta-llama/Meta-Llama-3-8B-Instruct`) |

## Project structure

```
src/gdelt_vector_stream/
  fetcher.py       # GDELT CSV parsing & event text generation
  embedder.py      # Sentence-Transformers embedding wrapper
  ingestor.py      # Pinecone batch-upsert with retry logic
  query.py         # Semantic search interface
  downloader.py    # Auto-download real GDELT data from master file list
  analyst.py       # RAG analyst: Pinecone retrieval + HF LLM
  trends.py        # Trending Topics Digest: multi-category world news briefing
  country_report.py# Country Intelligence Report: focused briefing for a specific country
  main.py          # Pipeline orchestration (fetch -> embed -> ingest)
api/
  server.py        # FastAPI backend
tests/
  test_pipeline.py # Unit tests for the core pipeline
  test_downloader.py
  test_trends.py   # Unit tests for the Trending Topics Digest
  test_country_report.py # Unit tests for the Country Intelligence Report
```

## Testing

```bash
pytest                                    # Full suite
pytest --cov=src --cov-report=term-missing  # With coverage
```

## License

MIT
