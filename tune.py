import os
import time
import subprocess
from pathlib import Path
import json

RESULTS_FILE = "tuning_results.md"

grid = [
    # Baseline
    {"CHUNK_SIZE": 512, "CHUNK_OVERLAP": 64, "RETRIEVAL_STRATEGY": "hybrid", "DIMENSION_STRATEGY": "coarse_then_full"},
    
    # Retrieval Strategies
    {"CHUNK_SIZE": 512, "CHUNK_OVERLAP": 64, "RETRIEVAL_STRATEGY": "vector_only", "DIMENSION_STRATEGY": "coarse_then_full"},
    {"CHUNK_SIZE": 512, "CHUNK_OVERLAP": 64, "RETRIEVAL_STRATEGY": "bm25_only", "DIMENSION_STRATEGY": "coarse_only"},
    
    # Coarse only (no rerank)
    {"CHUNK_SIZE": 512, "CHUNK_OVERLAP": 64, "RETRIEVAL_STRATEGY": "vector_only", "DIMENSION_STRATEGY": "coarse_only"},
    {"CHUNK_SIZE": 512, "CHUNK_OVERLAP": 64, "RETRIEVAL_STRATEGY": "hybrid", "DIMENSION_STRATEGY": "coarse_only"},

    # Different Chunking
    {"CHUNK_SIZE": 256, "CHUNK_OVERLAP": 32, "RETRIEVAL_STRATEGY": "hybrid", "DIMENSION_STRATEGY": "coarse_then_full"},
    {"CHUNK_SIZE": 1024, "CHUNK_OVERLAP": 128, "RETRIEVAL_STRATEGY": "hybrid", "DIMENSION_STRATEGY": "coarse_then_full"},
]

def write_header():
    with open(RESULTS_FILE, "w") as f:
        f.write("# Hyperparameter Tuning Results\n\n")
        f.write("| Chunk | Overlap | Retrieval | Dim Strategy | Ingest Time | Query Time | Avg Semantic | Avg Fuzzy | Avg Doc Score |\n")
        f.write("|-------|---------|-----------|--------------|-------------|------------|--------------|-----------|---------------|\n")

def run_experiment(config):
    env = os.environ.copy()
    for k, v in config.items():
        env[k] = str(v)
    
    print(f"\n[TUNE] Running: {config}")
    
    # Re-ingest
    print("   [TUNE] Re-ingesting...")
    t0 = time.perf_counter()
    res = subprocess.run(["python3", "fast_ingest.py", "./Testing Set"], env=env, capture_output=True, text=True)
    ingest_time = time.perf_counter() - t0
    if res.returncode != 0:
        print(f"Error ingesting: {res.stderr}")
        return
    
    # Query & Evaluate
    print("   [TUNE] Querying and Evaluating...")
    t0 = time.perf_counter()
    res = subprocess.run(["python3", "run_e2e_test.py", "--eval-only"], env=env, capture_output=True, text=True)
    query_time = time.perf_counter() - t0
    
    # We could parse stdout, but for simplicity let's just grep the averages or parse a JSON output if we had one.
    # Instead, let's adapt run_e2e_test to output a json summarize line: `__SUMMARY__:{...}`
    # Wait, we can just parse the console output.
    output = res.stdout
    avg_semantic, avg_fuzzy, avg_doc = 0.0, 0.0, 0.0
    
    for line in output.split("\n"):
        if line.startswith("AVG "):
            parts = line.split()
            if len(parts) >= 4:
                try:
                    avg_fuzzy = float(parts[1])
                    avg_semantic = float(parts[2])
                    avg_doc = float(parts[3])
                except ValueError:
                    pass

    # Write result
    with open(RESULTS_FILE, "a") as f:
        c = config
        f.write(f"| {c['CHUNK_SIZE']} | {c['CHUNK_OVERLAP']} | {c['RETRIEVAL_STRATEGY']} | {c['DIMENSION_STRATEGY']} "
                f"| {ingest_time:.1f}s | {query_time:.1f}s | **{avg_semantic:.3f}** | {avg_fuzzy:.3f} | {avg_doc:.3f} |\n")

if __name__ == "__main__":
    write_header()
    for cfg in grid:
        run_experiment(cfg)
    print(f"\n[TUNE] Finished! Results saved to {RESULTS_FILE}")
