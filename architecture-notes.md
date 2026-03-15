# Architecture Notes

## Current Pipeline Shape

### Ingestion
1. `pipeline.py` routes ingestion to `fast_ingest.py` when `FAST_INGEST=1`.
2. `fast_ingest.py` invokes the Go ingestor (`go-ingestor/main.go`) for parsing and chunking.
3. Go emits Arrow IPC data either:
	- file mode (`chunks.arrow`), or
	- streaming mode over Unix Domain Socket when `STREAMING_ENABLED=1`.
4. Python embeds chunks in `ingestion/embedder.py` and indexes in `storage/indexer.py`.

### Retrieval
1. Query path starts in `pipeline.py` and `retrieval/retriever.py`.
2. Hybrid retrieval uses:
	- dense vector candidates (Qdrant or FAISS),
	- BM25 keyword candidates,
	- cluster routing filter.
3. Merged candidates are reranked and passed to LLM answering in `query/answerer.py`.

## Active Defaults

1. Embedding model: `BAAI/bge-small-en-v1.5`.
2. Embedding dims: `EMBED_DIM_FULL=384`, `EMBED_DIM_COARSE=384`.
3. LLM default: `mistral-small:24b` via Ollama at `http://localhost:11434`.
4. Vector backend default: `qdrant` (with `faiss` available for benchmark runs).

## Recent Stability Notes

1. FAISS GPU retrieval now serializes search access to avoid allocator assertion failures under concurrent query load.
2. Streaming ingestion path uses Arrow IPC over UDS to avoid stdout data corruption.

## Operational Modes

1. Full run (decrypt + ingest + query): use `decrypt_and_run.py` with `SKIP_INGEST=0`.
2. Query-only run with existing indexes: use `decrypt_and_run.py` default `SKIP_INGEST=1`.
3. Backend override for dense retrieval/indexing:
	- `VECTOR_BACKEND=qdrant`
	- `VECTOR_BACKEND=faiss`

## Known Follow-ups

1. Submission endpoint path can vary by deployment; keep this configurable via env when possible.
2. Benchmark streaming vs batch with identical corpus and questions before making streaming the default.