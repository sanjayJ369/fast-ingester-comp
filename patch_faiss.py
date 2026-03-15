import sys
with open("storage/indexer.py") as f:
    code = f.read()

cpu_fallback = """
    faiss = _get_faiss()
    try:
        res = faiss.StandardGpuResources()
        use_gpu = True
    except AttributeError:
        use_gpu = False

    def build_and_save(arr, dim, out_path):
        import numpy as np
        arr = arr.astype(np.float32)
        faiss.normalize_L2(arr)
        cpu_index = faiss.IndexFlatIP(dim)
        if use_gpu:
            gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            gpu_index.add(arr)
            final_index = faiss.index_gpu_to_cpu(gpu_index)
        else:
            cpu_index.add(arr)
            final_index = cpu_index
        faiss.write_index(final_index, out_path)
"""

# Now we need to manually apply this logic if the original looks similar.
