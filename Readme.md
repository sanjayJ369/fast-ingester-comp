# Fast Corpus Ingestion

High-performance RAG pipeline for document ingestion and question answering. Combines a Go parser/chunker with Python embeddings and hybrid retrieval to ingest a corpus and answer 15 questions within a 30-second budget.

## Architecture

```
INGESTION
  Documents (PDF, DOCX, HTML, CSV, Excel, TXT)
       │
       ▼
  ┌──────────────┐     Arrow IPC      ┌──────────────────┐
  │  Go Ingestor │ ──────────────────► │  Python Pipeline  │
  │  Parse+Chunk │   chunks.arrow      │  Embed + Index    │
  └──────────────┘                     └────────┬─────────┘
                                                │
                         ┌──────────────────────┼──────────────────┐
                         ▼                      ▼                  ▼
                   Qdrant (384d)          BM25 Index         KMeans Clusters
                   Qdrant (128d)          bm25_index.pkl     cluster_map.json

QUERY (30s budget)
  Questions
       │
       ▼
  Embed query (384d + 128d)
       │
       ├─► Cluster routing → candidate doc_ids
       │
       ├─► Stage 1A: Coarse vector search (128d, top 50)
       ├─► Stage 1B: BM25 keyword search (top 20)
       │
       ▼
  Merge + deduplicate
       │
       ▼
  Stage 2: Full rerank (384d, top 5)
       │
       ▼
  LLM answer (Ollama or Claude) → JSON response
```

### Key Components

| Component | Tech | Purpose |
|-----------|------|---------|
| **Parser** | Go (`go-ingestor/parser/`) | Multi-format document extraction with concurrency |
| **Chunker** | Go (`go-ingestor/chunker/`) | Recursive splitting (512 chars, 64 overlap) with worker pool |
| **Arrow Writer** | Go (`go-ingestor/writer/`) | Zero-copy IPC handoff to Python via Apache Arrow |
| **Embedder** | Python (`ingestion/embedder.py`) | Snowflake Arctic Embed XS — dual output: 384d full + 128d coarse |
| **Indexer** | Python (`storage/indexer.py`) | Builds Qdrant collections, BM25 index, and KMeans clusters |
| **Retriever** | Python (`retrieval/retriever.py`) | Two-stage hybrid search with cluster-aware routing |
| **Answerer** | Python (`query/answerer.py`) | LLM generation via Ollama (local) or Claude (cloud) |

### Project Structure

```
fast-corpus-injestion/
├── config.py                # All tunable parameters
├── pipeline.py              # Main orchestrator (ingest + query)
├── fast_ingest.py           # Go+Arrow fast ingestion path
├── run_e2e_test.py          # E2E test with ground-truth scoring
├── test_pipeline.py         # Smoke test with synthetic docs
├── requirements.txt
├── Makefile
├── Dockerfile / Dockerfile.gpu
├── docker-compose.yml / docker-compose.e2e.yml
├── ingestion/
│   ├── parser.py            # Async multi-format doc parser
│   ├── chunker.py           # Character-level chunker
│   └── embedder.py          # Arctic-xs embedding (384+128 dims)
├── storage/
│   └── indexer.py           # Qdrant + BM25 + KMeans indexing
├── retrieval/
│   └── retriever.py         # Two-stage hybrid retrieval
├── query/
│   └── answerer.py          # Ollama / Claude answering
├── go-ingestor/
│   ├── main.go              # Go pipeline orchestrator
│   ├── parser/              # PDF, DOCX, HTML, CSV, Excel, TXT parsers
│   ├── chunker/             # Worker pool chunker
│   └── writer/              # Arrow IPC serializer
└── data/                    # Generated indexes and caches
```

## Setup on a Fresh Machine

### Prerequisites

- Python 3.10+
- Go 1.21+
- Docker (for Qdrant)
- `poppler-utils` (for PDF parsing via `pdftotext`)

### Quick Setup (Ubuntu GPU box)

```bash
make setup
```

This installs all system deps, Python/Go packages, pulls the Ollama model, and starts Qdrant in Docker.

### Manual Setup

**1. System dependencies**

```bash
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y \
    python3 python3-pip python3-venv \
    poppler-utils build-essential libffi-dev libarrow-dev

# macOS
brew install poppler apache-arrow go
```

**2. Python dependencies**

```bash
pip install -r requirements.txt

# For GPU acceleration (CUDA 12.4):
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

**3. Go ingestor**

```bash
make build
```

**4. Qdrant vector database**

```bash
docker run -d --name lucio_qdrant \
    -p 6333:6333 -p 6334:6334 \
    -v qdrant_data:/qdrant/storage \
    qdrant/qdrant:latest
```

**5. LLM backend (pick one)**

```bash
# Option A: Ollama (free, local)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral

# Option B: Claude API
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY
```

## Usage

### Ingest a corpus

```bash
# Fast path (Go parse/chunk + Python embed/index) — recommended
make ingest-fast CORPUS="./path/to/documents"

# Pure Python fallback
make ingest CORPUS="./path/to/documents"
```

### Query

```bash
make query QUESTIONS="questions.json"
```

### Run tests

```bash
# Smoke test with synthetic documents
make test

# End-to-end test with ground truth scoring
make e2e

# Benchmark ingestion timing
make bench
```

### Docker

```bash
# Standard (CPU)
docker compose up

# GPU + Ollama + Qdrant (E2E)
docker compose -f docker-compose.e2e.yml up
```

## Configuration

All parameters are in `config.py` and can be overridden via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `FAST_INGEST` | `0` | Set to `1` to use Go fast ingestion path |
| `LLM_BACKEND` | `ollama` | `ollama` or `anthropic` |
| `OLLAMA_MODEL` | `mistral` | Ollama model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `ANTHROPIC_API_KEY` | — | Required for Claude backend |
| `QDRANT_HOST` | `localhost` | Qdrant server host |

### Tuning Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `CHUNK_SIZE` | 512 | Characters per chunk |
| `CHUNK_OVERLAP` | 64 | Overlap between chunks |
| `EMBED_DIM_FULL` | 384 | Full embedding dimension |
| `EMBED_DIM_COARSE` | 128 | Coarse embedding dimension |
| `COARSE_TOP_K` | 50 | Stage 1 vector candidates |
| `BM25_TOP_K` | 20 | Stage 1 BM25 candidates |
| `FINAL_TOP_K` | 5 | Chunks sent to LLM |
| `N_CLUSTERS` | 15 | KMeans cluster count |
