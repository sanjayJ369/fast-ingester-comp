"""
storage/indexer.py — Build and persist all indexes.

Indexes built:
  1. Qdrant collection (full 768-dim)     → precise reranking
  2. Qdrant collection (coarse 128-dim)   → fast candidate retrieval
  3. BM25Okapi                            → keyword sparse retrieval
  4. Cluster map + adjacency graph        → corpus-aware routing
"""

import json
import pickle
import numpy as np
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, OptimizersConfigDiff
)
from rank_bm25 import BM25Okapi
from sklearn.cluster import MiniBatchKMeans

from ingestion.chunker import Chunk
from config import (
    QDRANT_HOST, QDRANT_PORT,
    QDRANT_COLLECTION_FULL, QDRANT_COLLECTION_COARSE,
    EMBED_DIM_FULL, EMBED_DIM_COARSE,
    N_CLUSTERS, CLUSTER_EXPAND,
    BM25_INDEX_PATH, CLUSTER_MAP_PATH, ADJ_GRAPH_PATH, CHUNKS_PATH,
    DATA_DIR,
)


# ── Qdrant client singleton ───────────────────────────────────────────────────

_qdrant: QdrantClient | None = None

def get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return _qdrant


# ── Qdrant indexing ───────────────────────────────────────────────────────────

def _ensure_collection(client: QdrantClient, name: str, dim: int):
    existing = [c.name for c in client.get_collections().collections]
    if name in existing:
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        optimizers_config=OptimizersConfigDiff(memmap_threshold=20000),
    )


def index_qdrant(
    chunks: list[Chunk],
    full_vecs: np.ndarray,
    coarse_vecs: np.ndarray,
) -> None:
    """
    Upsert all chunks into both Qdrant collections.
    Uses batch upserts of 256 points for speed.
    """
    client = get_qdrant()
    BATCH = 256

    for coll_name, vecs, dim in [
        (QDRANT_COLLECTION_FULL,   full_vecs,   EMBED_DIM_FULL),
        (QDRANT_COLLECTION_COARSE, coarse_vecs, EMBED_DIM_COARSE),
    ]:
        print(f"[QDRANT] Building {coll_name} ({dim}-dim)...")
        _ensure_collection(client, coll_name, dim)

        for i in range(0, len(chunks), BATCH):
            batch_chunks = chunks[i : i + BATCH]
            batch_vecs   = vecs[i : i + BATCH]

            points = [
                PointStruct(
                    id=idx + i,
                    vector=vec.tolist(),
                    payload={
                        "chunk_id":  c.chunk_id,
                        "doc_id":    c.doc_id,
                        "filename":  c.filename,
                        "text":      c.text,
                        "page_num":  c.page_num,
                        "chunk_idx": c.chunk_idx,
                    },
                )
                for idx, (c, vec) in enumerate(zip(batch_chunks, batch_vecs))
            ]
            client.upsert(collection_name=coll_name, points=points)

        print(f"[QDRANT] ✅ {coll_name} — {len(chunks)} points indexed")


# ── BM25 indexing ─────────────────────────────────────────────────────────────

def index_bm25(chunks: list[Chunk]) -> BM25Okapi:
    """
    Build BM25 index over all chunks.
    Tokenizes on whitespace (fast; good enough for retrieval).
    """
    print("[BM25] Building keyword index...")
    tokenized = [c.text.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)

    Path(DATA_DIR).mkdir(exist_ok=True)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)

    print(f"[BM25] ✅ Index saved → {BM25_INDEX_PATH}")
    return bm25


# ── Cluster map (corpus-aware routing) ───────────────────────────────────────

def build_cluster_map(
    chunks: list[Chunk],
    coarse_vecs: np.ndarray,
) -> tuple[MiniBatchKMeans, dict]:
    """
    Cluster all chunk vectors into N_CLUSTERS groups using MiniBatchKMeans.
    Builds:
      - cluster_map: {doc_id: [cluster_ids]}
      - adjacency_graph: {doc_id: [related_doc_ids]} (shared cluster = edge)

    Both saved to disk as JSON.
    """
    actual_clusters = min(N_CLUSTERS, len(chunks))
    print(f"[CLUSTER] Clustering {len(chunks)} chunks into {actual_clusters} clusters...")
    kmeans = MiniBatchKMeans(
        n_clusters=actual_clusters,
        random_state=42,
        batch_size=1024,
        n_init=3,
    )
    labels = kmeans.fit_predict(coarse_vecs)

    # doc → set of clusters it appears in
    doc_clusters: dict[str, set[int]] = {}
    chunk_cluster_map: dict[str, int] = {}

    for chunk, label in zip(chunks, labels):
        doc_clusters.setdefault(chunk.doc_id, set()).add(int(label))
        chunk_cluster_map[chunk.chunk_id] = int(label)

    # cluster → list of doc_ids
    cluster_docs: dict[int, list[str]] = {}
    for doc_id, clusters in doc_clusters.items():
        for c in clusters:
            cluster_docs.setdefault(c, []).append(doc_id)

    # adjacency: docs that share at least one cluster are "related"
    adjacency: dict[str, list[str]] = {d: [] for d in doc_clusters}
    for doc_id, clusters in doc_clusters.items():
        related: set[str] = set()
        for c in clusters:
            related.update(cluster_docs.get(c, []))
        related.discard(doc_id)
        adjacency[doc_id] = list(related)

    # serialise
    cluster_map_data = {
        "doc_clusters":       {k: list(v) for k, v in doc_clusters.items()},
        "cluster_docs":       {str(k): v for k, v in cluster_docs.items()},
        "chunk_cluster_map":  chunk_cluster_map,
    }

    Path(DATA_DIR).mkdir(exist_ok=True)
    with open(CLUSTER_MAP_PATH, "w") as f:
        json.dump(cluster_map_data, f)
    with open(ADJ_GRAPH_PATH, "w") as f:
        json.dump(adjacency, f)

    print(f"[CLUSTER] ✅ Cluster map → {CLUSTER_MAP_PATH}")
    print(f"[CLUSTER] ✅ Adjacency graph → {ADJ_GRAPH_PATH}")

    return kmeans, cluster_map_data


# ── Chunk cache ───────────────────────────────────────────────────────────────

def save_chunks(chunks: list[Chunk]) -> None:
    Path(DATA_DIR).mkdir(exist_ok=True)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print(f"[STORAGE] ✅ {len(chunks)} chunks cached → {CHUNKS_PATH}")


def load_chunks() -> list[Chunk]:
    with open(CHUNKS_PATH, "rb") as f:
        return pickle.load(f)
