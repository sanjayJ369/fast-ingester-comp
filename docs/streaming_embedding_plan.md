# Streaming Embedding Migration Plan

## Goal
Move ingestion from a batch handoff model to a bounded streaming pipeline so parsing, chunking, embedding, and vector indexing overlap instead of running as isolated stages.

## Scope
- In scope:
  - Streaming Go to Python handoff for chunk batches via Unix Domain Sockets (UDS).
  - Deterministic streamed chunk ordering and ID assignment.
  - Bounded backpressure across parse, chunk, embed, and index stages.
  - Append-oriented vector indexing for both supported dense backends (Qdrant/FAISS).
  - Benchmarking and retrieval parity checks against the current fast path.
  - **Infrastructure:** Targeted for Ubuntu/Debian on `vast.ai` GPU boxes.
- Out of scope for phase 1:
  - Moving embeddings fully into Go.
  - Replacing BM25 or cluster-map retrieval components.
  - Redesigning the query pipeline.

## Current Baseline
The current fast path is:
Go parse + chunk -> Arrow file on disk -> Python read-all -> embedding -> vector index -> BM25 -> cluster map

Key current bottleneck:
- Python only starts embedding after Go has finished writing the full Arrow file.
- `go-ingestor` collects all raw chunks before assigning IDs and writing.

## Target Architecture
The target streaming fast path is:
file discovery -> parse workers -> chunk sequencer -> Arrow stream (UDS) -> Python embedding re-batcher -> vector index writer -> finalization (BM25/Cluster map)

### Design Constraints & Refinements
1.  **Transport:** Use Arrow IPC streaming over a **Unix Domain Socket (UDS)**. 
    *   Python (consumer) creates a temporary socket file (e.g., `/tmp/lucio_ingest.sock`).
    *   Go (producer) connects to the socket and streams Arrow batches.
    *   **Benefit:** Keeps `stdout` and `stderr` completely free for human-readable logging and progress bars without risk of stream corruption.
2.  **Backpressure:** Leverage UDS buffer semantics. Python's `asyncio` consumer will naturally apply backpressure to the Go producer if the embedding queue is full.
3.  **Deterministic IDs:** Ensure `chunk_id` generation (hashing/UUID) is identical between batch and stream paths to maintain parity.
4.  **Re-batching Layer:** Python will implement a bounded queue to re-batch incoming Arrow chunks into optimal GPU inference sizes (e.g., 128/256) independently of Go's transport batch size.
5.  **Finalization Manifest:** Python will maintain an append-only in-memory or temporary manifest of all chunk metadata to build the BM25 index and Cluster map after the socket closes.

## Implementation Plan

### Phase 1: Lock the rollout shape & Infrastructure
1.  Add `STREAMING_ENABLED=1` flag support to `pipeline.py` and `fast_ingest.py`.
2.  Update `Makefile` with a new `e2e-stream-faiss` target that drives `run_e2e_test.py --ingest` with streaming enabled.
3.  **VastAI Verification:** Verify `libarrow-dev` (apt) and `apache-arrow-go` (go.mod) version compatibility on the production GPU box.

### Phase 2: Make chunk IDs stream-safe
1.  Introduce a document-aware sequencer in Go that buffers out-of-order pages and emits chunks in deterministic `page_num` / `page_order`.
2.  Assign `chunk_idx` and `chunk_id` immediately upon sequencing.
3.  **Test:** `go-ingestor/chunker/sequencer_test.go` proving parity with batch ordering.

### Phase 3: Replace Arrow file handoff with Arrow streaming (Go side)
1.  Add a streaming Arrow writer in Go that can write to a `net.Conn` (UDS).
2.  Add a `--socket-path` flag to `go-ingestor` to trigger streaming mode.
3.  Ensure all logs use `stderr` or the standard `stdout` (now safe since data is in UDS).

### Phase 4: Python streaming consumer and re-batcher
1.  Update `fast_ingest.py` to create the UDS server, spawn the Go process with the `--socket-path` flag, and accept the connection.
2.  Implement the Python-side `asyncio` re-batching queue to feed the `Embedder`.
3.  Manage socket lifecycle: create, listen, and ensure cleanup of the `/tmp/*.sock` file on exit or failure.

### Phase 5: Append-only Vector Indexing
1.  For FAISS: Use an in-process appendable builder.
2.  For Qdrant: Perform batch upserts as re-batched vectors become available.

### Phase 6: Finalization & Metadata
1.  Accumulate chunk metadata (text, doc_name, page_num) in an append-only manifest during the stream.
2.  After the stream closes, trigger the build of BM25 and Cluster map from this manifest.

### Phase 7: Backpressure and Failure Handling
1.  Use bounded channels in Go and `asyncio.Queue` in Python.
2.  Implement structured failure propagation: if Go fails, Python should cleanup partial artifacts and exit.

## Verification
1.  **Parity:** Run `make e2e-stream-faiss` and ensure similarity scores and document matches are identical to `make e2e-faiss`.
2.  **Performance:** Record "Time to first embedding" and "Total ingestion time".
3.  **Stability:** Verify clean socket cleanup and log readability on `vast.ai`.

## Related Docs
- [docs/faiss_migration_plan.md](docs/faiss_migration_plan.md)
- [Makefile](../Makefile)
- [run_e2e_test.py](../run_e2e_test.py)
