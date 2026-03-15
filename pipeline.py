"""
pipeline.py — Main orchestrator for the Lucio Challenge pipeline.

Two entry points:
  1. ingest(doc_dir)  — parse, chunk, embed, index (run before challenge clock)
  2. query(questions) — retrieve + answer all 15 questions (the 30s window)

Usage:
  # Step 1: ingest your corpus (do this before the clock starts!)
  python pipeline.py ingest ./data/corpus

  # Step 2: answer questions
  python pipeline.py query "What is X?" "Who did Y?" ...
"""

import asyncio
import json
import time
from pathlib import Path

from ingestion.parser import parse_all
from ingestion.chunker import chunk_all
from ingestion.embedder import embed_chunks, get_model
from storage.indexer import (
    index_qdrant, index_bm25, build_cluster_map, save_chunks
)
from retrieval.retriever import retrieve_all
from query.answerer import answer_all


# ── PHASE 1: Ingestion ────────────────────────────────────────────────────────

async def ingest(doc_dir: str) -> dict:
    """
    Full ingestion pipeline.

    If FAST_INGEST=1 env var is set, uses the Go-based fast path
    (Go parse/chunk → Arrow IPC → Python embed/index).

    Otherwise falls back to the original all-Python pipeline:
    1. Parse all documents in parallel
    2. Chunk all documents
    3. Embed all chunks (MRL: full + coarse)
    4. Index into Qdrant (both collections)
    5. Build BM25 index
    6. Build cluster map + adjacency graph
    7. Cache chunks to disk

    Returns timing breakdown dict.
    """
    import os
    if os.getenv("FAST_INGEST", "0") == "1":
        from fast_ingest import fast_ingest
        return await fast_ingest(doc_dir)
    t_total = time.perf_counter()
    timings = {}

    # discover all files
    corpus_path = Path(doc_dir)
    filepaths = [
        p for p in corpus_path.rglob("*")
        if p.is_file() and p.suffix.lower() in
           {".pdf", ".docx", ".doc", ".txt", ".html", ".htm", ".csv", ".xlsx", ".xls"}
    ]
    print(f"\n{'='*60}")
    print(f"[PIPELINE] 📂 Found {len(filepaths)} documents in {doc_dir}")
    print(f"{'='*60}\n")

    # 1. Parse
    t0 = time.perf_counter()
    docs = await parse_all(filepaths)
    timings["parse_s"] = round(time.perf_counter() - t0, 2)
    print(f"[PIPELINE] ⏱  Parse: {timings['parse_s']}s\n")

    # 2. Chunk
    t0 = time.perf_counter()
    chunks = chunk_all(docs)
    timings["chunk_s"] = round(time.perf_counter() - t0, 2)
    print(f"[PIPELINE] ⏱  Chunk: {timings['chunk_s']}s\n")

    # 3. Embed (loads model once, then batches)
    get_model()  # warm up model before timing
    t0 = time.perf_counter()
    full_vecs, coarse_vecs = embed_chunks(chunks)
    timings["embed_s"] = round(time.perf_counter() - t0, 2)
    print(f"[PIPELINE] ⏱  Embed: {timings['embed_s']}s\n")

    # 4. Qdrant indexing
    t0 = time.perf_counter()
    from config import VECTOR_BACKEND
    if VECTOR_BACKEND == "faiss":
        from storage.indexer import index_faiss
        index_faiss(chunks, full_vecs, coarse_vecs)
    else:
        index_qdrant(chunks, full_vecs, coarse_vecs)
    timings["qdrant_s"] = round(time.perf_counter() - t0, 2)
    print(f"[PIPELINE] ⏱  Vector Index: {timings['qdrant_s']}s\n")

    # 5. BM25
    t0 = time.perf_counter()
    index_bm25(chunks)
    timings["bm25_s"] = round(time.perf_counter() - t0, 2)
    print(f"[PIPELINE] ⏱  BM25: {timings['bm25_s']}s\n")

    # 6. Cluster map
    t0 = time.perf_counter()
    kmeans, cluster_data = build_cluster_map(chunks, coarse_vecs)
    # store centroids in cluster_map for query-time routing
    centroids = kmeans.cluster_centers_.tolist()
    cluster_data["centroids"] = centroids
    from config import CLUSTER_MAP_PATH
    with open(CLUSTER_MAP_PATH, "w") as f:
        json.dump(cluster_data, f)
    timings["cluster_s"] = round(time.perf_counter() - t0, 2)
    print(f"[PIPELINE] ⏱  Cluster: {timings['cluster_s']}s\n")

    # 7. Cache chunks
    save_chunks(chunks)

    timings["total_s"] = round(time.perf_counter() - t_total, 2)

    print(f"\n{'='*60}")
    print(f"[PIPELINE] ✅ INGESTION COMPLETE in {timings['total_s']}s")
    print(f"  Documents : {len(docs)}")
    print(f"  Chunks    : {len(chunks)}")
    print(f"  Timings   : {timings}")
    print(f"{'='*60}\n")

    return timings


# ── PHASE 2: Query ────────────────────────────────────────────────────────────

async def query(questions: list[str]) -> list[dict]:
    """
    Full query pipeline — this is what runs inside the 30-second window.

    1. Retrieve top chunks for all 15 questions (parallel)
    2. Generate answers for all 15 questions (parallel LLM calls)
    3. Return structured answer list

    Returns list of answer dicts ready for submission to Lucio server.
    """
    t_total = time.perf_counter()

    print(f"\n{'='*60}")
    print(f"[PIPELINE] 🚀 QUERY PHASE — {len(questions)} questions")
    print(f"{'='*60}\n")

    # 1. Parallel retrieval
    t0 = time.perf_counter()
    retrieved = await retrieve_all(questions)
    t_retrieve = round(time.perf_counter() - t0, 2)
    print(f"[PIPELINE] ⏱  Retrieval: {t_retrieve}s")
    for i, chunks in enumerate(retrieved):
        print(f"  Q{i+1:02d}: {len(chunks)} chunks retrieved")

    # 2. Parallel LLM answering
    t0 = time.perf_counter()
    answers = await answer_all(questions, retrieved)
    t_llm = round(time.perf_counter() - t0, 2)
    print(f"\n[PIPELINE] ⏱  LLM answering: {t_llm}s")

    # 3. Format for submission
    results = []
    for q, a in zip(questions, answers):
        results.append({
            "question":     a.question,
            "answer":       a.answer,
            "doc_name":     a.doc_name,
            "page_numbers": a.page_numbers,
            "confidence":   a.confidence,
        })

    t_total_s = round(time.perf_counter() - t_total, 2)

    print(f"\n{'='*60}")
    print(f"[PIPELINE] ✅ QUERY COMPLETE in {t_total_s}s")
    print(f"  Retrieve : {t_retrieve}s")
    print(f"  LLM      : {t_llm}s")
    print(f"  Total    : {t_total_s}s")
    if t_total_s < 30:
        print(f"  🏆 WITHIN 30s BUDGET ({30 - t_total_s:.1f}s to spare)")
    else:
        print(f"  ⚠️  OVER BUDGET by {t_total_s - 30:.1f}s — optimise!")
    print(f"{'='*60}\n")

    return results


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python pipeline.py ingest <doc_dir>")
        print("  python pipeline.py query <q1> <q2> ... <q15>")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "ingest":
        doc_dir = sys.argv[2] if len(sys.argv) > 2 else "./data/corpus"
        asyncio.run(ingest(doc_dir))

    elif mode == "query":
        questions = sys.argv[2:]
        if not questions:
            # demo questions for testing
            questions = [
                "What is the main topic of the corpus?",
                "Who are the key authors mentioned?",
            ]
        results = asyncio.run(query(questions))
        print(json.dumps(results, indent=2))

    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)
