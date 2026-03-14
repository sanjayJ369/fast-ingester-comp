.PHONY: build clean ingest ingest-fast query test setup e2e bench

GO_DIR       = go-ingestor
GO_BIN       = $(GO_DIR)/ingestor
CORPUS       ?= ./Testing Set
OLLAMA_MODEL ?= mistral

# Build Go ingestor binary
build:
	cd $(GO_DIR) && CGO_LDFLAGS="-L$(CURDIR)/$(GO_DIR)/lib" go build -o ingestor .

# Fast ingestion (Go parse/chunk + Python embed/index)
ingest-fast: build
	FAST_INGEST=1 python pipeline.py ingest "$(CORPUS)"

# Original Python-only ingestion (fallback)
ingest:
	python pipeline.py ingest "$(CORPUS)"

# Run fast_ingest.py standalone
ingest-standalone: build
	python fast_ingest.py "$(CORPUS)"

# Query phase
query:
	python pipeline.py query $(QUESTIONS)

# Clean build artifacts
clean:
	rm -f $(GO_BIN) data/chunks.arrow

# Run test pipeline
test: build
	python test_pipeline.py

# ── Fresh Ubuntu GPU box setup + E2E ─────────────────────────────────────

# Install everything on a fresh Ubuntu box
setup:
	sudo apt-get update && sudo apt-get install -y \
		python3 python3-pip python3-venv \
		poppler-utils build-essential libffi-dev \
		ca-certificates curl gnupg
	curl -fsSL https://apache.jfrog.io/artifactory/arrow/$(lsb_release --id --short | tr 'A-Z' 'a-z')/apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb -o /tmp/arrow.deb
	sudo apt-get install -y /tmp/arrow.deb
	sudo apt-get update && sudo apt-get install -y libarrow-dev
	sudo snap install go --classic
	pip install --break-system-packages -r requirements.txt einops
	pip install --break-system-packages torch --index-url https://download.pytorch.org/whl/cu124
	curl -fsSL https://ollama.com/install.sh | sh
	ollama pull $(OLLAMA_MODEL)
	docker run -d --name lucio_qdrant -p 6333:6333 -p 6334:6334 \
		-v qdrant_data:/qdrant/storage qdrant/qdrant:latest || true
	cd $(GO_DIR) && CGO_LDFLAGS="-L$(CURDIR)/$(GO_DIR)/lib" go build -o ingestor .

# Run e2e test (assumes make setup was run)
e2e: build
	python run_e2e_test.py --ingest

# Run ingestion only and print benchmarks
bench: build
	python fast_ingest.py "$(CORPUS)"
