"""
fast_ingest.py — Fast ingestion using Go parser/chunker + Python embedder/indexer.

The Go binary reads all documents in parallel (goroutines + pdftotext),
chunks the text, and writes an Arrow IPC file. This script reads that
file via pyarrow (zero-copy), then embeds and indexes using the existing
Python infrastructure.

Usage:
    # Build Go binary first:  cd go-ingestor && go build -o ingestor .
    python fast_ingest.py ./data/corpus
    python fast_ingest.py "./Testing Set"
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import pyarrow.ipc as ipc

from ingestion.chunker import Chunk
from ingestion.embedder import embed_chunks, get_model
from storage.indexer import index_qdrant, index_bm25, build_cluster_map, save_chunks
from config import CLUSTER_MAP_PATH, DATA_DIR

ARROW_PATH = "./data/chunks.arrow"
GO_BINARY = "./go-ingestor/ingestor"


def run_go_ingestor(doc_dir: str) -> float:
    """Run the Go ingestor binary. Returns elapsed seconds."""
    t0 = time.perf_counter()
    result = subprocess.run(
        [GO_BINARY, doc_dir, ARROW_PATH],
        capture_output=True, text=True,
    )
    elapsed = time.perf_counter() - t0

    # Print Go output
    if result.stdout:
        print(result.stdout, end="")
    if result.returncode != 0:
        print(f"[GO] STDERR: {result.stderr}")
        raise RuntimeError(f"Go ingestor failed with code {result.returncode}")

    return elapsed


def arrow_to_chunks(arrow_path: str) -> list[Chunk]:
    """Read Arrow IPC file and convert to list of Chunk objects (zero-copy read)."""
    reader = ipc.open_file(arrow_path)
    table = reader.read_all()

    # Convert columnar Arrow to row-based Chunk objects
    chunk_ids = table.column("chunk_id")
    doc_ids = table.column("doc_id")
    filenames = table.column("filename")
    texts = table.column("text")
    page_nums = table.column("page_num")
    chunk_idxs = table.column("chunk_idx")

    chunks = []
    for i in range(table.num_rows):
        try:
            text = texts[i].as_py()
        except UnicodeDecodeError:
            # Handle non-UTF-8 bytes from PDF extraction
            text = texts[i].as_buffer().to_pybytes().decode("utf-8", errors="replace")
        if not text or not text.strip():
            continue
        chunks.append(Chunk(
            chunk_id=chunk_ids[i].as_py(),
            doc_id=doc_ids[i].as_py(),
            filename=filenames[i].as_py(),
            text=text,
            page_num=page_nums[i].as_py(),
            chunk_idx=chunk_idxs[i].as_py(),
        ))
    return chunks


def fast_ingest(doc_dir: str) -> dict:
    """Full fast ingestion pipeline."""
    timings = {}
    t_total = time.perf_counter()

    Path(DATA_DIR).mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[FAST] Fast ingestion: {doc_dir}")
    print(f"{'='*60}\n")

    # 1. Go: parse + chunk → Arrow IPC
    timings["go_parse_chunk_s"] = round(run_go_ingestor(doc_dir), 2)
    print(f"[FAST] Go parse+chunk: {timings['go_parse_chunk_s']}s\n")

    # 2. Arrow → Chunk objects
    t0 = time.perf_counter()
    chunks = arrow_to_chunks(ARROW_PATH)
    timings["arrow_read_s"] = round(time.perf_counter() - t0, 2)
    print(f"[FAST] Arrow read: {timings['arrow_read_s']}s ({len(chunks)} chunks)\n")

    # 3. Embed (Python, MPS-accelerated)
    get_model()  # warm up model before timing
    t0 = time.perf_counter()
    full_vecs, coarse_vecs = embed_chunks(chunks)
    timings["embed_s"] = round(time.perf_counter() - t0, 2)
    print(f"[FAST] Embed: {timings['embed_s']}s\n")

    # 4. Qdrant indexing
    t0 = time.perf_counter()
    index_qdrant(chunks, full_vecs, coarse_vecs)
    timings["qdrant_s"] = round(time.perf_counter() - t0, 2)
    print(f"[FAST] Qdrant: {timings['qdrant_s']}s\n")

    # 5. BM25
    t0 = time.perf_counter()
    index_bm25(chunks)
    timings["bm25_s"] = round(time.perf_counter() - t0, 2)
    print(f"[FAST] BM25: {timings['bm25_s']}s\n")

    # 6. Cluster map
    t0 = time.perf_counter()
    kmeans, cluster_data = build_cluster_map(chunks, coarse_vecs)
    cluster_data["centroids"] = kmeans.cluster_centers_.tolist()
    with open(CLUSTER_MAP_PATH, "w") as f:
        json.dump(cluster_data, f)
    timings["cluster_s"] = round(time.perf_counter() - t0, 2)
    print(f"[FAST] Cluster: {timings['cluster_s']}s\n")

    # 7. Cache chunks
    save_chunks(chunks)

    timings["total_s"] = round(time.perf_counter() - t_total, 2)

    print(f"\n{'='*60}")
    print(f"[FAST] INGESTION COMPLETE in {timings['total_s']}s")
    print(f"  Chunks    : {len(chunks)}")
    print(f"  Timings   : {timings}")
    print(f"{'='*60}\n")

    return timings


if __name__ == "__main__":
    doc_dir = sys.argv[1] if len(sys.argv) > 1 else "./data/corpus"
    fast_ingest(doc_dir)
