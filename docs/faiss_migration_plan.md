# FAISS GPU Migration Plan (No faiss-cpu)

## Goal
Adopt FAISS as a GPU-only dense retrieval backend for benchmark runs, while preserving current Qdrant behavior as fallback and avoiding `faiss-cpu` installation in shared environments.

## Scope
- In scope:
  - GPU-only FAISS dependency path.
  - Backend selection via config/env.
  - Dense retrieval migration (coarse + rerank path).
  - Benchmark parity checks (latency + retrieval quality).
- Out of scope:
  - Removing Qdrant immediately.
  - ANN index tuning (IVF/HNSW/PQ) in phase 1.

## Current Baseline
- Dense indexing and retrieval are Qdrant-based in [storage/indexer.py](storage/indexer.py#L58) and [retrieval/retriever.py](retrieval/retriever.py#L102).
- BM25 stays in [storage/indexer.py](storage/indexer.py#L103) and [retrieval/retriever.py](retrieval/retriever.py#L124).
- Cluster routing currently uses MiniBatchKMeans in [storage/indexer.py](storage/indexer.py#L122).
- GPU image path is [Dockerfile.gpu](Dockerfile.gpu#L14).
- Bare-metal GPU setup path is [Makefile](Makefile#L30).

## Design Decisions
1. Do not add FAISS to [requirements.txt](requirements.txt).
2. Install FAISS only in GPU-specific setup paths:
   - [Dockerfile.gpu](Dockerfile.gpu#L14)
   - [Makefile](Makefile#L30)
3. Add `VECTOR_BACKEND` switch:
   - `qdrant` (default)
   - `faiss` (GPU benchmark path)
4. Keep BM25 unchanged for hybrid retrieval.
5. Start with exact FAISS indexes:
   - `IndexFlatIP(128)` for coarse stage.
   - `IndexFlatIP(384)` for full-stage rerank/scoring.
6. Reassess cluster routing after FAISS benchmark data; keep initially for parity.

## Implementation Plan

### Phase 1: Dependency and Backend Plumbing
1. Add backend config in [config.py](config.py):
   - `VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "qdrant")`
2. Add FAISS install in [Dockerfile.gpu](Dockerfile.gpu#L14) only.
3. Add FAISS install in [Makefile](Makefile#L30) `setup` target only.
4. Add GPU benchmark targets in [Makefile](Makefile#L73):
   - `bench-faiss`
   - `e2e-faiss`
   - export `VECTOR_BACKEND=faiss` for these targets.

### Phase 2: FAISS Index Build + Persistence
1. Introduce FAISS index build functions in [storage/indexer.py](storage/indexer.py#L58):
   - build coarse and full indexes from normalized vectors.
2. Persist artifacts under `data/`:
   - `coarse.faiss`
   - `full.faiss` (optional if rerank is numpy-only over candidate vectors)
   - metadata map (row_id -> chunk metadata).
3. Keep existing BM25 and chunk cache unchanged.

### Phase 3: Retrieval Path Integration
1. In [retrieval/retriever.py](retrieval/retriever.py#L102), route `_coarse_search` by backend:
   - Qdrant path (existing)
   - FAISS path (new)
2. In [retrieval/retriever.py](retrieval/retriever.py#L163), route `_rerank` by backend:
   - Qdrant path (existing)
   - FAISS/numpy candidate scoring path (new)
3. Keep query embedding and BM25 flow unchanged in [retrieval/retriever.py](retrieval/retriever.py#L207).

### Phase 4: Benchmarks and Acceptance
1. Compare `qdrant` vs `faiss` on same corpus/questions:
   - ingestion index build time
   - query latency for all 15 questions
   - top-k overlap and final answer quality metrics
2. Acceptance criteria:
   - FAISS backend installs only in GPU paths.
   - No `faiss-cpu` anywhere.
   - FAISS benchmark latency is equal or better than Qdrant.
   - No quality regression beyond agreed threshold.
3. If FAISS wins, make `faiss` default only in GPU benchmark workflows.

## Risks and Mitigations
1. FAISS wheel/CUDA compatibility:
   - Pin versions and validate in both Docker and bare-metal setup.
2. Divergence between Docker and Makefile setups:
   - Use same install source/version in both files.
3. Metadata mapping bugs:
   - Add integrity checks for row_id/chunk_id consistency.
4. Cluster routing complexity:
   - Keep behind flag, disable if no measurable gain.

## Suggested Milestones
1. M1 (0.5 day): dependency plumbing + backend flag.
2. M2 (1 day): FAISS indexing + persistence.
3. M3 (1 day): retrieval integration + parity tests.
4. M4 (0.5 day): benchmark report + decision.

## Definition of Done
1. GPU-only FAISS path operational in [Dockerfile.gpu](Dockerfile.gpu#L14) and [Makefile](Makefile#L30).
2. `VECTOR_BACKEND=faiss` runs ingestion and query successfully.
3. Benchmarks documented with Qdrant comparison.
4. Default non-GPU/local paths remain unaffected.
