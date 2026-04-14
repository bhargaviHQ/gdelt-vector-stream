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
python -m gdelt_vector_stream.downloader --max-files 10 --sample-size 100

# Watch mode: poll every 15 minutes for new files
python -m gdelt_vector_stream.downloader --watch
```

### Semantic search

```bash
python -m gdelt_vector_stream.query --text "protests in Southeast Asia"
```

### RAG analyst

Ask natural language questions grounded in your GDELT data:

```bash
python -m gdelt_vector_stream.analyst "What conflicts are happening in the Middle East?"

# Show retrieved events alongside the answer
python -m gdelt_vector_stream.analyst "Trade tensions between US and China" --show-context

# Use a different model
python -m gdelt_vector_stream.analyst "Climate events in Europe" --model Qwen/Qwen2.5-7B-Instruct
```

### API server

A FastAPI backend is available for building frontends:

```bash
pip install fastapi uvicorn
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
  main.py          # Pipeline orchestration (fetch -> embed -> ingest)
api/
  server.py        # FastAPI backend
tests/
  test_pipeline.py # Unit tests for the core pipeline
  test_downloader.py
```

## Testing

```bash
pytest                                    # Full suite
pytest --cov=src --cov-report=term-missing  # With coverage
```

## License

MIT
