"""
ingestion/embedder.py — Embedding module.

Uses BAAI/bge-small-en-v1.5 (33M params, 384-dim).
Produces TWO embedding vectors per chunk:
  - full   (384-dim): used for precise stage-2 reranking
  - coarse (384-dim): also 384-dim, no truncation for BGE-small (it's fast enough)

On Apple M4, sentence-transformers will use the MPS backend automatically.
"""

import asyncio
import threading
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from ingestion.chunker import Chunk
from config import (
    EMBED_MODEL, EMBED_DIM_FULL, EMBED_DIM_COARSE, EMBED_BATCH_SIZE,
    EMBED_QUERY_PREFIX, EMBED_DOC_PREFIX,
)


# ── Model singleton ───────────────────────────────────────────────────────────

_model: SentenceTransformer | None = None
_model_lock = threading.Lock()

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[EMBEDDER] Loading {EMBED_MODEL} in float16 ...")
        _model = SentenceTransformer(
            EMBED_MODEL, 
            trust_remote_code=True,
            model_kwargs={"torch_dtype": torch.float16}
        )
        print(f"[EMBEDDER] ✅ Model loaded ({EMBED_DIM_FULL}-dim)")
    return _model


# ── Core embedding ────────────────────────────────────────────────────────────

def _embed_batch(texts: list[str]) -> np.ndarray:
    """
    Embed a batch of texts. Returns float32 array of shape (N, EMBED_DIM_FULL).
    """
    model = get_model()
    # Documentation suggests no specific prefix for docs, just raw text.
    vecs = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return vecs.astype(np.float32)


def _embed_query(query: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Embed a single query. Returns (full_vec, coarse_vec).
    Uses a lock for thread safety during concurrent query embedding.
    """
    model = get_model()
    # BGE-small-en-v1.5 instruction format: "{instruction}{query}"
    prompt = f"{EMBED_QUERY_PREFIX}{query}"
    
    with _model_lock:
        vec = model.encode(
            prompt,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

    full   = vec[:EMBED_DIM_FULL]
    coarse = vec[:EMBED_DIM_COARSE]
    # re-normalize coarse after truncation (if dimensions are different)
    if EMBED_DIM_FULL != EMBED_DIM_COARSE:
        coarse = coarse / (np.linalg.norm(coarse) + 1e-9)
    return full, coarse


# ── Batch embed all chunks ────────────────────────────────────────────────────

def embed_chunks(chunks: list[Chunk]) -> tuple[np.ndarray, np.ndarray]:
    """
    Embed all chunks in batches.

    Returns:
        full_vecs   — shape (N, EMBED_DIM_FULL)
        coarse_vecs — shape (N, EMBED_DIM_COARSE)
    """
    texts = [c.text for c in chunks]
    n = len(texts)
    print(f"[EMBEDDER] Embedding {n} chunks in batches of {EMBED_BATCH_SIZE}...")

    all_vecs: list[np.ndarray] = []
    for i in range(0, n, EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        vecs  = _embed_batch(batch)
        all_vecs.append(vecs)
        if (i // EMBED_BATCH_SIZE) % 5 == 0:
            print(f"[EMBEDDER] ... {min(i + EMBED_BATCH_SIZE, n)}/{n}")

    full_vecs = np.vstack(all_vecs)                        # (N, EMBED_DIM_FULL)
    coarse_vecs = full_vecs[:, :EMBED_DIM_COARSE].copy()   # (N, EMBED_DIM_COARSE)

    # re-normalize coarse vectors after truncation (if dimensions differ)
    if EMBED_DIM_FULL != EMBED_DIM_COARSE:
        norms = np.linalg.norm(coarse_vecs, axis=1, keepdims=True)
        coarse_vecs = coarse_vecs / (norms + 1e-9)

    print(f"[EMBEDDER] ✅ Embeddings ready — "
          f"full {full_vecs.shape}, coarse {coarse_vecs.shape}")
    return full_vecs, coarse_vecs


# ── Query embedding (used at query time, async-safe) ─────────────────────────

async def embed_query_async(query: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Async wrapper for query embedding so it doesn't block the event loop
    during the query phase when questions fire concurrently.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _embed_query, query)
