"""
retrieval/retriever.py — Two-stage hybrid retriever with cluster routing.

Per question, the pipeline is:

  1. Embed question (coarse 128-dim + full 384-dim)
  2. Route to nearest cluster(s) → restrict search space
  3. Stage 1A: Coarse 128-dim vector search → top COARSE_TOP_K candidates
  4. Stage 1B: BM25 keyword search → top BM25_TOP_K candidates
  5. Merge & deduplicate candidates from 1A + 1B
  6. Stage 2: Full 384-dim cosine rerank on merged set → top FINAL_TOP_K

All 15 questions run concurrently via asyncio.gather().

Backend selection
-----------------
Set the VECTOR_BACKEND env var (or config.py constant) to:
  "qdrant" (default) — uses Qdrant for stages 1A and 2.
  "faiss"            — uses GPU FAISS for stage 1A; numpy dot-product for stage 2.
"""

import asyncio
import json
import pickle
import numpy as np
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny
from rank_bm25 import BM25Okapi

from ingestion.chunker import Chunk
from ingestion.embedder import embed_query_async
from storage.indexer import get_qdrant
from config import (
    QDRANT_COLLECTION_FULL, QDRANT_COLLECTION_COARSE,
    COARSE_TOP_K, BM25_TOP_K, FINAL_TOP_K, CLUSTER_EXPAND,
    BM25_INDEX_PATH, CLUSTER_MAP_PATH, CHUNKS_PATH,
    FAISS_COARSE_PATH, FAISS_FULL_PATH, FAISS_META_PATH,
    VECTOR_BACKEND,
)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    chunk_id:  str
    doc_id:    str
    filename:  str
    text:      str
    page_num:  int
    score:     float


# ── Index loading (cached singletons) ────────────────────────────────────────

_bm25:        BM25Okapi    | None = None
_chunks:      list[Chunk]  | None = None
_cluster_map: dict         | None = None

# FAISS singletons — loaded once, GPU-placed, held for process lifetime
_faiss_coarse = None
_faiss_meta:  dict | None = None  # {int_row_id: chunk_metadata_dict}
_gpu_res      = None              # faiss.StandardGpuResources


import threading

_index_lock = threading.Lock()
_faiss_lock = threading.Lock()
# GPU FAISS search is not thread-safe with shared StandardGpuResources.
# Serialize search calls to prevent allocator stack corruption.
_faiss_search_lock = threading.Lock()

def _load_indexes():
    global _bm25, _chunks, _cluster_map
    with _index_lock:
        if _bm25 is None:
            with open(BM25_INDEX_PATH, "rb") as f:
                _bm25 = pickle.load(f)

        if _chunks is None:
            with open(CHUNKS_PATH, "rb") as f:
                _chunks = pickle.load(f)

        if _cluster_map is None:
            with open(CLUSTER_MAP_PATH, "r") as f:
                _cluster_map = json.load(f)


def _load_faiss_indexes() -> None:
    """Load FAISS indexes from disk and GPU-place them.  Called once per process."""
    global _faiss_coarse, _faiss_meta, _gpu_res

    with _faiss_lock:
        try:
            import faiss  # noqa: PLC0415
        except ImportError:
            raise RuntimeError(
                "FAISS is not installed.  "
                "Set VECTOR_BACKEND=qdrant or install: pip install faiss-gpu-cu12"
            ) from None

        if _gpu_res is None:
            _gpu_res = faiss.StandardGpuResources()

        if _faiss_coarse is None:
            cpu_idx = faiss.read_index(FAISS_COARSE_PATH)
            _faiss_coarse = faiss.index_cpu_to_gpu(_gpu_res, 0, cpu_idx)

        if _faiss_meta is None:
            with open(FAISS_META_PATH, "rb") as f:
                _faiss_meta = pickle.load(f)


# ── Cluster routing ───────────────────────────────────────────────────────────

def _get_candidate_doc_ids(
    coarse_query_vec: np.ndarray,
    cluster_map: dict,
) -> list[str] | None:
    """
    Find the CLUSTER_EXPAND nearest cluster centroids to the query vector.
    Returns doc_ids that belong to those clusters, or None to search all.

    We store centroid info in the cluster_map during ingestion (added below).
    If centroids aren't available, fall back to full search.
    """
    centroids = cluster_map.get("centroids")
    if centroids is None:
        return None     # fall back: search all docs

    centroid_arr = np.array(centroids, dtype=np.float32)
    # cosine sim = dot product (vectors are normalized)
    sims = centroid_arr @ coarse_query_vec
    top_clusters = np.argsort(sims)[::-1][:CLUSTER_EXPAND].tolist()

    cluster_docs = cluster_map.get("cluster_docs", {})
    candidate_doc_ids: set[str] = set()
    for c in top_clusters:
        candidate_doc_ids.update(cluster_docs.get(str(c), []))

    return list(candidate_doc_ids) if candidate_doc_ids else None


# ── Stage 1A: Coarse vector search ────────────────────────────────────────────

def _coarse_search_qdrant(
    client: QdrantClient,
    coarse_vec: np.ndarray,
    candidate_doc_ids: list[str] | None,
) -> list[dict]:
    """Qdrant path: search 128-dim collection for top COARSE_TOP_K candidates."""
    search_filter = None
    if candidate_doc_ids:
        search_filter = Filter(
            must=[FieldCondition(
                key="doc_id",
                match=MatchAny(any=candidate_doc_ids),
            )]
        )

    result = client.query_points(
        collection_name=QDRANT_COLLECTION_COARSE,
        query=coarse_vec.tolist(),
        limit=COARSE_TOP_K,
        query_filter=search_filter,
        with_payload=True,
    )
    return [h.payload for h in result.points]


def _coarse_search_faiss(
    coarse_vec: np.ndarray,
    candidate_doc_ids: list[str] | None,
) -> list[dict]:
    """
    FAISS path: search 128-dim GPU index for top COARSE_TOP_K candidates.
    Filters by doc_id after retrieval (FAISS has no payload filter).
    """
    _load_faiss_indexes()
    vec = np.ascontiguousarray(coarse_vec, dtype=np.float32).reshape(1, -1)

    # Fetch more than COARSE_TOP_K when filtering so we can trim afterwards.
    k = COARSE_TOP_K * 4 if candidate_doc_ids else COARSE_TOP_K
    with _faiss_search_lock:
        _, indices = _faiss_coarse.search(vec, k)

    candidate_set = set(candidate_doc_ids) if candidate_doc_ids else None
    results: list[dict] = []
    for row_id in indices[0]:
        if row_id < 0:
            continue  # FAISS pads with -1 when fewer than k results exist
        meta = _faiss_meta[int(row_id)]
        if candidate_set and meta["doc_id"] not in candidate_set:
            continue
        results.append(meta)
        if len(results) == COARSE_TOP_K:
            break
    return results


def _coarse_search(
    client: QdrantClient,
    coarse_vec: np.ndarray,
    candidate_doc_ids: list[str] | None,
) -> list[dict]:
    """Dispatch coarse search to the active backend."""
    if VECTOR_BACKEND == "faiss":
        return _coarse_search_faiss(coarse_vec, candidate_doc_ids)
    return _coarse_search_qdrant(client, coarse_vec, candidate_doc_ids)


# ── Stage 1B: BM25 keyword search ─────────────────────────────────────────────

def _bm25_search(
    query: str,
    chunks: list[Chunk],
    bm25: BM25Okapi,
    candidate_doc_ids: list[str] | None,
) -> list[dict]:
    """Score all chunks with BM25, filter by candidates, return top BM25_TOP_K."""
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)

    indexed = list(enumerate(scores))
    if candidate_doc_ids:
        doc_set = set(candidate_doc_ids)
        indexed = [(i, s) for i, s in indexed if chunks[i].doc_id in doc_set]

    top = sorted(indexed, key=lambda x: x[1], reverse=True)[:BM25_TOP_K]

    results = []
    for idx, score in top:
        if score > 0:
            c = chunks[idx]
            results.append({
                "chunk_id": c.chunk_id,
                "doc_id":   c.doc_id,
                "filename": c.filename,
                "text":     c.text,
                "page_num": c.page_num,
                "_bm25_score": float(score),
            })
    return results


# ── Stage 2: Full 768-dim rerank ──────────────────────────────────────────────

def _rerank(
    client: QdrantClient,
    full_query_vec: np.ndarray,
    candidates: list[dict],
) -> list[RetrievedChunk]:
    """
    Rerank merged candidates using full 768-dim cosine similarity.
    Fetches precise vectors from Qdrant for the candidate chunk_ids.
    """
    if not candidates:
        return []

    # get chunk_ids for lookup
    chunk_ids = [c["chunk_id"] for c in candidates]

    # search precise collection, filtered to just these chunks
    result = client.query_points(
        collection_name=QDRANT_COLLECTION_FULL,
        query=full_query_vec.tolist(),
        limit=FINAL_TOP_K,
        query_filter=Filter(
            must=[FieldCondition(
                key="chunk_id",
                match=MatchAny(any=chunk_ids),
            )]
        ),
        with_payload=True,
    )

    return [
        RetrievedChunk(
            chunk_id = h.payload["chunk_id"],
            doc_id   = h.payload["doc_id"],
            filename = h.payload["filename"],
            text     = h.payload["text"],
            page_num = h.payload["page_num"],
            score    = h.score,
        )
        for h in result.points
    ]


# ── Full retrieval pipeline (single question) ─────────────────────────────────

async def retrieve(query: str) -> list[RetrievedChunk]:
    """
    Full two-stage hybrid retrieval for a single question.
    Called concurrently for all 15 questions.
    """
    _load_indexes()
    client = get_qdrant()

    # 1. Embed query (async, non-blocking)
    full_vec, coarse_vec = await embed_query_async(query)

    # 2. Cluster routing
    candidate_doc_ids = _get_candidate_doc_ids(coarse_vec, _cluster_map)

    # 3. Stage 1A + 1B in parallel (both are sync but fast — run in executor)
    loop = asyncio.get_running_loop()

    coarse_task = loop.run_in_executor(
        None, _coarse_search, client, coarse_vec, candidate_doc_ids
    )
    bm25_task = loop.run_in_executor(
        None, _bm25_search, query, _chunks, _bm25, candidate_doc_ids
    )

    coarse_results, bm25_results = await asyncio.gather(coarse_task, bm25_task)

    # 4. Merge + deduplicate
    seen: set[str] = set()
    merged: list[dict] = []
    for r in coarse_results + bm25_results:
        if r["chunk_id"] not in seen:
            seen.add(r["chunk_id"])
            merged.append(r)

    # 5. Stage 2: full rerank
    final = await loop.run_in_executor(
        None, _rerank, client, full_vec, merged
    )

    return final


# ── Batch retrieval (all 15 questions at once) ────────────────────────────────

async def retrieve_all(questions: list[str]) -> list[list[RetrievedChunk]]:
    """
    Fire all questions concurrently.
    Returns a list of results aligned with the input questions.
    """
    tasks = [retrieve(q) for q in questions]
    results = await asyncio.gather(*tasks)
    return list(results)
