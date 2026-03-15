.PHONY: build clean ingest ingest-fast ingest-stream query test setup fix-gpu e2e bench bench-faiss e2e-faiss e2e-stream-faiss

GO_DIR       = go-ingestor
GO_BIN       = $(GO_DIR)/ingestor
PYTHON       ?= python3
PIP          ?= pip3
CORPUS       ?= ./Testing Set
OLLAMA_MODEL ?= mistral

# Build Go ingestor binary
build:
	cd $(GO_DIR) && CGO_LDFLAGS="-L$(CURDIR)/$(GO_DIR)/lib" go build -buildvcs=false -o ingestor .

# Fast ingestion (Go parse/chunk + Python embed/index)
ingest-fast: build
	FAST_INGEST=1 $(PYTHON) pipeline.py ingest "$(CORPUS)"

# Streaming ingestion (Go parse/chunk -> UDS -> Python embed/index)
ingest-stream: build
	FAST_INGEST=1 STREAMING_ENABLED=1 $(PYTHON) pipeline.py ingest "$(CORPUS)"

# Original Python-only ingestion (fallback)
ingest:
	$(PYTHON) pipeline.py ingest "$(CORPUS)"

# Run fast_ingest.py standalone
ingest-standalone: build
	$(PYTHON) fast_ingest.py "$(CORPUS)"

# Query phase
query:
	$(PYTHON) pipeline.py query $(QUESTIONS)

# Clean build artifacts
clean:
	rm -f $(GO_BIN) data/chunks.arrow

# Run test pipeline
test: build
	$(PYTHON) test_pipeline.py

# ── Fresh Ubuntu GPU box setup + E2E ─────────────────────────────────────

# Install everything on a fresh Ubuntu box (tested on Ubuntu 22.04 / VastAI)
setup:
	@echo "==> Installing system packages..."
	sudo apt-get update && sudo apt-get install -y \
		python3 python3-pip python3-venv \
		poppler-utils build-essential libffi-dev \
		ca-certificates curl gnupg lsb-release
	@echo "==> Installing Apache Arrow..."
	curl -fsSL "https://apache.jfrog.io/artifactory/arrow/$$(lsb_release --id --short | tr 'A-Z' 'a-z')/apache-arrow-apt-source-latest-$$(lsb_release --codename --short).deb" -o /tmp/arrow.deb
	sudo apt-get install -y /tmp/arrow.deb
	sudo apt-get update && sudo apt-get install -y libarrow-dev
	@echo "==> Installing Go..."
	sudo snap install go --classic || echo "snap not available, install Go manually"
	@echo "==> Installing Python dependencies (torch first for CUDA)..."
	$(PIP) install torch --index-url https://download.pytorch.org/whl/cu124
	$(PIP) install -r requirements.txt einops
	@echo "==> Installing GPU-only FAISS (not in requirements.txt)..."
	$(PIP) install faiss-gpu-cu12
	@echo "==> Installing Ollama..."
	curl -fsSL https://ollama.com/install.sh | sh
	ollama pull $(OLLAMA_MODEL)
	@echo "==> Starting Qdrant..."
	docker run -d --name lucio_qdrant -p 6333:6333 -p 6334:6334 \
		-v qdrant_data:/qdrant/storage qdrant/qdrant:latest || true
	@echo "==> Downloading tokenizers library..."
	mkdir -p $(GO_DIR)/lib
	curl -fsSL https://github.com/daulet/tokenizers/releases/download/v1.26.0/libtokenizers.linux-amd64.tar.gz \
		| tar -xz -C $(GO_DIR)/lib
	@echo "==> Building Go ingestor..."
	cd $(GO_DIR) && CGO_LDFLAGS="-L$(CURDIR)/$(GO_DIR)/lib" go build -o ingestor .
	@echo "==> Fixing GPU driver (if needed)..."
	$(MAKE) fix-gpu || true
	@echo "==> Setup complete. Run 'make bench' to test."

# Fix NVIDIA driver/library version mismatch (common on VastAI / cloud GPU boxes)
fix-gpu:
	@echo "==> Reloading NVIDIA kernel modules..."
	sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia 2>/dev/null || true
	sudo modprobe nvidia
	sudo modprobe nvidia_uvm
	sudo modprobe nvidia_drm
	nvidia-smi
	$(PYTHON) -c "import torch; assert torch.cuda.is_available(), 'CUDA still not available'; print('CUDA OK:', torch.cuda.get_device_name(0))"

# Run e2e test (assumes make setup was run)
e2e: build
	$(PYTHON) run_e2e_test.py --ingest

# Run ingestion only and print benchmarks
bench: build
	$(PYTHON) fast_ingest.py "$(CORPUS)"

# GPU benchmark: FAISS backend ingestion only
bench-faiss: build
	@echo "==> Benchmarking FAISS GPU backend ingestion..."
	VECTOR_BACKEND=faiss $(PYTHON) fast_ingest.py "$(CORPUS)"

# GPU benchmark: FAISS backend full e2e
e2e-faiss: build
	@echo "==> Running e2e with FAISS GPU backend..."
	VECTOR_BACKEND=faiss $(PYTHON) run_e2e_test.py --ingest

# GPU benchmark: FAISS backend full e2e with streaming
e2e-stream-faiss: build
	@echo "==> Running e2e with FAISS GPU backend (STREAMING)..."
	VECTOR_BACKEND=faiss STREAMING_ENABLED=1 $(PYTHON) run_e2e_test.py --ingest
