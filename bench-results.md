# Benchmark Results

**Date:** 2026-03-15
**Machine:** VastAI GPU box (root@153.198.41.69:50532)
**OS:** Ubuntu 22.04 (jammy), 18 CPUs
**GPU:** NVIDIA GeForce RTX 3090 (24GB), CUDA 13.0, Driver 580.126.09
**Corpus:** Testing Set (282 MB, 18 files)

## Run 3 — FAISS GPU backend (Optimised Ingestion)

| Stage           | Time (s) |
|-----------------|----------|
| Go parse+chunk  | 5.35     |
| Arrow read      | 0.16     |
| Embed           | 8.91     |
| FAISS index     | 0.50     |
| BM25 index      | 0.63     |
| Cluster         | 0.54     |
| **Total**       | **19.44** |

- **Chunks produced:** 18,217
- **Embedding model:** Snowflake/snowflake-arctic-embed-xs (384-dim)
- **Batch size:** 256
- **FAISS Speedup:** 29.5x vs Qdrant (14.75s -> 0.50s)
- **Total Ingestion Speedup:** 1.76x vs Qdrant run (34.2s -> 19.4s)

### E2E Query Performance (10 questions)
- **Retrieval :** 1.40s (140ms/q)
- **LLM (Ollama):** 10.30s (1.03s/q)
- **Total E2E   :** 11.70s (Well within 30s budget)
- **Avg Semantic:** 0.843

## Run 2 — GPU enabled (Qdrant Backend)

| Stage           | Time (s) |
|-----------------|----------|
| Go parse+chunk  | 5.44     |
| Arrow read      | 0.17     |
| Embed           | 8.76     |
| Qdrant index    | 14.75    |
| BM25 index      | 0.65     |
| Cluster         | 0.45     |
| **Total**       | **34.23** |

- **Date:** 2026-03-14
- **Chunks produced:** 18,215
- **Embedding model:** Snowflake/snowflake-arctic-embed-xs (384-dim)
- **Batch size:** 256
- **Embed speedup vs CPU:** 17.4x (152s -> 8.8s)
- **Bottleneck:** Qdrant indexing (14.75s = 43% of total)

## Run 1 — CPU fallback (broken CUDA driver)

| Stage           | Time (s) |
|-----------------|----------|
| Go parse+chunk  | 5.37     |
| Arrow read      | 0.17     |
| Embed           | 152.17   |
| Qdrant index    | 14.45    |
| BM25 index      | 0.63     |
| Cluster         | 0.54     |
| **Total**       | **188.83** |

- **Date:** 2026-03-14
- CUDA had a driver/library version mismatch (kernel module 580.95 vs userspace 580.126)
- Embeddings fell back to CPU

## Notes

- Fix: `make fix-gpu` reloads nvidia kernel modules to resolve driver mismatch
- ~~Bottleneck is now Qdrant indexing (14.75s = 43% of total)~~ (Resolved by FAISS)
- Go ingestor parses 18 files (incl. 541-page and 403-page PDFs) in ~5.4s with parallel page splitting
- ~~Qdrant indexes both full (384-dim) and coarse (128-dim) collections~~ (FAISS now handles both on GPU)
