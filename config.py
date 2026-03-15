"""
config.py — Central config for the Lucio pipeline.
All tunable knobs in one place.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM ──────────────────────────────────────────────────────────────────────
LLM_BACKEND       = os.getenv("LLM_BACKEND", "ollama")  # "ollama" or "anthropic"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL         = "claude-sonnet-4-20250514"           # for anthropic backend
LLM_MAX_TOKENS    = 512
LLM_TEMPERATURE   = 0.0          # deterministic for factual QA

# ── Ollama ───────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL      = os.getenv("OLLAMA_MODEL", "qwen3.5:9b")

# ── Embedding ─────────────────────────────────────────────────────────────────
EMBED_MODEL        = "qwen3-embedding:8b"
EMBED_DIM_FULL     = 384    # precise index
EMBED_DIM_COARSE   = 128    # fast coarse index (truncation)
EMBED_BATCH_SIZE   = 256    # chunks per batch — arctic-xs is tiny, crank it up
EMBED_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
EMBED_DOC_PREFIX   = ""     # arctic-embed uses no doc prefix

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE         = 512    # characters
CHUNK_OVERLAP      = 64

# ── Retrieval ─────────────────────────────────────────────────────────────────
COARSE_TOP_K       = 50     # stage 1: coarse vector search candidates
BM25_TOP_K         = 20     # stage 1: BM25 keyword candidates
FINAL_TOP_K        = 5      # stage 2: reranked chunks sent to LLM

# ── Clustering ────────────────────────────────────────────────────────────────
N_CLUSTERS         = 15     # one cluster per question (heuristic)
CLUSTER_EXPAND     = 2      # also query N nearest clusters for safety

# ── Vector backend ───────────────────────────────────────────────────────────
# "qdrant" (default, all environments)
# "faiss"  (GPU benchmark path only — requires faiss-gpu-cu12)
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "qdrant")

# ── Qdrant ────────────────────────────────────────────────────────────────────
QDRANT_HOST              = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT              = 6333
QDRANT_COLLECTION_FULL   = "lucio_full"     # 384-dim
QDRANT_COLLECTION_COARSE = "lucio_coarse"   # 128-dim

# ── Storage paths ─────────────────────────────────────────────────────────────
DATA_DIR          = "./data"
BM25_INDEX_PATH   = "./data/bm25_index.pkl"
CLUSTER_MAP_PATH  = "./data/cluster_map.json"
ADJ_GRAPH_PATH    = "./data/adjacency_graph.json"
CHUNKS_PATH       = "./data/chunks.pkl"       # raw chunks cache
ARROW_CHUNKS_PATH = "./data/chunks.arrow"     # Arrow IPC from Go ingestor

# ── FAISS artifact paths (GPU benchmark path only) ────────────────────────────
FAISS_COARSE_PATH = "./data/coarse.faiss"    # 128-dim IndexFlatIP on disk
FAISS_FULL_PATH   = "./data/full.faiss"      # 384-dim IndexFlatIP on disk
FAISS_META_PATH   = "./data/faiss_meta.pkl"  # {int_row_id: chunk_metadata_dict}
