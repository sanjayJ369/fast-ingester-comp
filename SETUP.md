# GPU Box Setup Guide

Step-by-step record of deploying this pipeline on a fresh VastAI Ubuntu 22.04 GPU box.

## Quick Start

```bash
ssh -i ~/.ssh/vastAI -p <PORT> root@<HOST> -L 8080:localhost:8080
git clone https://github.com/sanjayJ369/fast-ingester-comp.git
cd fast-ingester-comp
make setup
make bench
```

## What `make setup` Does

1. Installs system packages (python3, pip, poppler, build tools, lsb-release)
2. Adds the Apache Arrow apt repo and installs `libarrow-dev`
3. Installs Go via snap
4. Installs PyTorch with CUDA 12.4 support (installed first so sentence-transformers picks it up)
5. Installs Python requirements from `requirements.txt`
6. Installs Ollama and pulls the model
7. Starts Qdrant in Docker
8. Downloads the HuggingFace tokenizers C library into `go-ingestor/lib/`
9. Builds the Go ingestor binary
10. Runs `make fix-gpu` to reload NVIDIA kernel modules

## Issues We Hit and How They Were Fixed

### 1. `lsb_release` not found — Arrow .deb URL was empty

**Symptom:**
```
curl: (22) The requested URL returned error: 404
```
The Arrow download URL uses `lsb_release --id --short` and `lsb_release --codename --short` to build the distro-specific URL. On some Ubuntu boxes `lsb-release` isn't installed by default.

**Fix:** Added `lsb-release` to the apt install list. Also quoted the full URL to prevent word-splitting issues with Make's `$$()` subshell syntax.

### 2. `go-ingestor/lib/` directory didn't exist

**Symptom:**
```
tar: go-ingestor/lib: Cannot open: No such file or directory
```
The tokenizers tarball was being extracted into a directory that hadn't been created.

**Fix:** Added `mkdir -p $(GO_DIR)/lib` before the `curl | tar` command.

### 3. `python` not found (only `python3` exists)

**Symptom:**
```
/bin/sh: 1: python: not found
```
Ubuntu 22.04 doesn't ship a `python` symlink, only `python3`.

**Fix:** Added a `PYTHON ?= python3` and `PIP ?= pip3` variable to the Makefile, replaced all `python` / `pip` calls with `$(PYTHON)` / `$(PIP)`.

### 4. CUDA not working — NVIDIA driver/library version mismatch

**Symptom:**
```
Failed to initialize NVML: Driver/library version mismatch
NVML library version: 580.126
```
```python
CUDA initialization: CUDA unknown error
torch.cuda.is_available() => False
```
The loaded kernel module was v580.95.05 but the userspace library was v580.126.09. This is common on VastAI boxes where the driver gets updated but the kernel module isn't reloaded.

**Diagnosis:**
```bash
# Check kernel module version
cat /proc/driver/nvidia/version
# => NVIDIA UNIX Open Kernel Module ... 580.95.05

# Check userspace library version
dpkg -l | grep nvidia-driver
# => nvidia-driver-580-server-open  580.126.09

# Check module info
modinfo nvidia | head -5
# => version: 580.126.09  (correct .ko file exists, just not loaded)
```

**Fix:**
```bash
# Unload old modules (safe if no GPU processes running)
sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia

# Load the updated modules
sudo modprobe nvidia
sudo modprobe nvidia_uvm
sudo modprobe nvidia_drm

# Verify
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available())"  # True
```

This is now automated as `make fix-gpu`.

## Benchmark Results

### With GPU (RTX 3090) — after fix

| Stage           | Time (s) |
|-----------------|----------|
| Go parse+chunk  | 5.44     |
| Arrow read      | 0.17     |
| Embed           | 8.76     |
| Qdrant index    | 14.75    |
| BM25 index      | 0.65     |
| Cluster         | 0.45     |
| **Total**       | **34.23** |

### Without GPU (CPU fallback) — before fix

| Stage           | Time (s) |
|-----------------|----------|
| Go parse+chunk  | 5.37     |
| Arrow read      | 0.17     |
| Embed           | 152.17   |
| Qdrant index    | 14.45    |
| BM25 index      | 0.63     |
| Cluster         | 0.54     |
| **Total**       | **188.83** |

Embedding went from 152s (CPU) to 8.8s (GPU) — **17.4x speedup**.
Total pipeline went from 189s to 34s — **5.5x speedup**.

## Useful Commands

```bash
make setup       # Full install on fresh box
make fix-gpu     # Reload NVIDIA modules if CUDA isn't working
make bench       # Run ingestion benchmark
make e2e         # Run full end-to-end test
make clean       # Remove build artifacts
```
