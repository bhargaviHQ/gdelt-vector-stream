# gdelt-vector-stream

[![CI](https://github.com/bhargaviHQ/gdelt-vector-stream/actions/workflows/ci.yml/badge.svg)](https://github.com/bhargaviHQ/gdelt-vector-stream/actions/workflows/ci.yml)
[![GDELT Ingest](https://github.com/bhargaviHQ/gdelt-vector-stream/actions/workflows/gdelt-ingest.yml/badge.svg)](https://github.com/bhargaviHQ/gdelt-vector-stream/actions/workflows/gdelt-ingest.yml)

A real-time geographic sentiment tracker using GDELT Global Knowledge Graph and Pinecone indexing.

## CI / CD

| Workflow | Trigger | Purpose |
|---|---|---|
| **CI** (`.github/workflows/ci.yml`) | Every push & pull request to `main`/`master` | Runs `ruff` linting and `pytest` unit tests on Python 3.11 & 3.12 |
| **GDELT Ingest** (`.github/workflows/gdelt-ingest.yml`) | Every **5 hours** via cron (`0 */5 * * *`), plus manual dispatch | Queries the GDELT master manifest, downloads the latest `.export.CSV.zip` files, embeds them, and upserts vectors to Pinecone. Commits the updated `data/.processed_files.json` tracker back to the repo so each run skips already-processed files. |

### Required repository secrets

Add these in **Settings → Secrets and variables → Actions** before running the ingest workflow:

| Secret | Description |
|---|---|
| `PINECONE_API_KEY` | Your Pinecone API key |
| `PINECONE_INDEX_NAME` | Name of the Pinecone index to upsert vectors into |

### Manual ingest trigger

You can kick off an ingest run at any time from the **Actions** tab → *GDELT Ingest* → **Run workflow**.  
Optional inputs:

- **sample_size** — records to sample per file (default `100`)
- **max_files** — how many new GDELT files to process (default `3`)
- **dry_run** — fetch & parse without writing to Pinecone (for testing)
