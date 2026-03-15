import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

from ingestion.chunker import Chunk
from ingestion.embedder import embed_chunks, get_model, _embed_batch
from storage.indexer import index_qdrant, index_bm25, build_cluster_map, save_chunks, StreamingIndexer
from config import CLUSTER_MAP_PATH, DATA_DIR, EMBED_BATCH_SIZE, EMBED_DIM_COARSE, VECTOR_BACKEND

ARROW_PATH = "./data/chunks.arrow"
GO_BINARY = "./go-ingestor/ingestor"


class StreamingRebatcher:
    """
    Receives individual chunks, re-batches them for optimal GPU inference,
    and pushes to the StreamingIndexer.
    """

    def __init__(self, indexer: StreamingIndexer, batch_size: int = EMBED_BATCH_SIZE):
        self.indexer = indexer
        self.batch_size = batch_size
        self.buffer: list[Chunk] = []
        self.processed_count = 0
        self.lock = asyncio.Lock()

    async def add_chunks(self, chunks: list[Chunk]):
        async with self.lock:
            self.buffer.extend(chunks)
            while len(self.buffer) >= self.batch_size:
                batch = self.buffer[:self.batch_size]
                self.buffer = self.buffer[self.batch_size:]
                await self._process_batch(batch)

    async def flush(self):
        async with self.lock:
            if self.buffer:
                await self._process_batch(self.buffer)
                self.buffer = []

    async def _process_batch(self, batch: list[Chunk]):
        # Run embedding in executor to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        texts = [c.text for c in batch]

        # We use the existing _embed_batch logic
        full_vecs = await loop.run_in_executor(None, _embed_batch, texts)

        # Post-process (coarse truncation + re-normalization)
        coarse_vecs = full_vecs[:, :EMBED_DIM_COARSE].copy()
        norms = np.linalg.norm(coarse_vecs, axis=1, keepdims=True)
        coarse_vecs = coarse_vecs / (norms + 1e-9)

        # Append to indexer
        self.indexer.append_batch(batch, full_vecs, coarse_vecs)
        self.processed_count += len(batch)
        print(f"[STREAM] Embedded and indexed {self.processed_count} chunks...")


class PipeReceiver:
    """
    Receives Arrow IPC stream from a file-like object (e.g. process stdout).
    """

    def __init__(self, stream_file, rebatcher: StreamingRebatcher, loop: asyncio.AbstractEventLoop):
        self.stream_file = stream_file
        self.rebatcher = rebatcher
        self.loop = loop
        self.done_event = asyncio.Event()

    async def start(self):
        print(f"[STREAM] Starting Arrow stream reader from pipe...")
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._read_stream)
            print("[STREAM] Pipe reader finished")
        except Exception as e:
            print(f"[STREAM] Pipe Error: {e}")
        finally:
            self.done_event.set()

    def _read_stream(self):
        """Synchronous reading loop for Arrow stream."""
        try:
            with pa.ipc.open_stream(self.stream_file) as stream_reader:
                print(f"[STREAM] Arrow stream opened with schema: {stream_reader.schema}")
                for batch in stream_reader:
                    # Convert Arrow batch to list of Chunks
                    chunks = self._batch_to_chunks(batch)
                    # We need to call back into the async loop
                    fut = asyncio.run_coroutine_threadsafe(
                        self.rebatcher.add_chunks(chunks),
                        self.loop,
                    )
                    # Propagate async failures immediately and preserve backpressure.
                    fut.result()
            print(f"[STREAM] Arrow stream EOF reached")
        except EOFError:
            print(f"[STREAM] Arrow stream EOFError")
        except Exception as e:
            print(f"[STREAM] _read_stream error: {e}")
            import traceback
            traceback.print_exc()

    def _batch_to_chunks(self, batch: pa.RecordBatch) -> list[Chunk]:
        data = batch.to_pydict()
        chunks = []
        for i in range(batch.num_rows):
            chunks.append(Chunk(
                chunk_id=data["chunk_id"][i],
                doc_id=data["doc_id"][i],
                filename=data["filename"][i],
                text=data["text"][i],
                page_num=data["page_num"][i],
                chunk_idx=data["chunk_idx"][i],
            ))
        return chunks


async def streaming_ingest_async(doc_dir: str) -> dict:
    """Async orchestration for streaming ingestion."""
    t0 = time.perf_counter()
    timings = {}

    Path(DATA_DIR).mkdir(exist_ok=True)

    indexer = StreamingIndexer(vector_backend=os.getenv("VECTOR_BACKEND", "qdrant"))
    rebatcher = StreamingRebatcher(indexer)
    loop = asyncio.get_running_loop()

    # Start Go process
    print(f"[STREAM] Spawning Go ingestor: {GO_BINARY} --stream {doc_dir}")
    # We MUST use stdout=PIPE to capture the Arrow stream
    go_proc = await asyncio.create_subprocess_exec(
        GO_BINARY, "--stream", doc_dir,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    # Wrap the stdout PIPE with a PipeReceiver
    # We need a synchronous file-like object from the reader
    # This is tricky with asyncio.StreamReader.
    # Better: use a real OS pipe or a temporary file.
    # OR: use a small helper to bridge StreamReader -> Sync read()

    class SyncReaderBridge:
        def __init__(self, reader, loop):
            self.reader = reader
            self.loop = loop
            self.closed = False
        def read(self, n):
            if self.closed:
                return b""
            fut = asyncio.run_coroutine_threadsafe(self.reader.read(n), self.loop)
            return fut.result()
        def seekable(self):
            return False
        def close(self):
            self.closed = True

    receiver = PipeReceiver(SyncReaderBridge(go_proc.stdout, loop), rebatcher, loop)
    receiver_task = asyncio.create_task(receiver.start())

    # Stream logs from Go stderr
    async def log_stream(stream, prefix):
        while True:
            line = await stream.readline()
            if not line:
                break
            print(f"{prefix} {line.decode().strip()}")

    log_task_err = asyncio.create_task(log_stream(go_proc.stderr, "[GO]"))

    # Wait for Go to finish
    return_code = await go_proc.wait()
    if return_code != 0:
        print(f"[STREAM] Go process failed with return code {return_code}")
        receiver_task.cancel()
        raise RuntimeError("Go ingestor failed")

    # Wait for receiver to finish
    await receiver_task

    # Final flush
    await rebatcher.flush()

    # Finalize indexer
    t_finalize = time.perf_counter()
    indexer.finalize()
    timings["finalize_s"] = round(time.perf_counter() - t_finalize, 2)

    timings["total_s"] = round(time.perf_counter() - t0, 2)
    return timings


async def streaming_ingest(doc_dir: str) -> dict:
    """
    Fast ingestion with streaming (stdout pipe).
    Go parse+chunk -> stdout -> Python embed+index (overlap)
    """
    return await streaming_ingest_async(doc_dir)


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


def fast_ingest_batch(doc_dir: str) -> dict:
    """Original batch fast ingestion pipeline."""
    timings = {}
    t_total = time.perf_counter()

    Path(DATA_DIR).mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[FAST] Batch fast ingestion: {doc_dir}")
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

    # 4. Vector Indexing
    t0 = time.perf_counter()
    from config import VECTOR_BACKEND
    if VECTOR_BACKEND == "faiss":
        from storage.indexer import index_faiss
        index_faiss(chunks, full_vecs, coarse_vecs)
    else:
        index_qdrant(chunks, full_vecs, coarse_vecs)
    timings["vector_index_s"] = round(time.perf_counter() - t0, 2)
    print(f"[FAST] Vector Index: {timings['vector_index_s']}s\n")

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


async def fast_ingest(doc_dir: str) -> dict:
    """Entry point for fast ingestion, branches to batch or streaming."""
    if os.getenv("STREAMING_ENABLED", "0") == "1":
        return await streaming_ingest(doc_dir)
    else:
        # Run synchronous batch ingest in executor to be async-friendly
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, fast_ingest_batch, doc_dir)


if __name__ == "__main__":
    doc_dir = sys.argv[1] if len(sys.argv) > 1 else "./Testing Set"
    asyncio.run(fast_ingest(doc_dir))
