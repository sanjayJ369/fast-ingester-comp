.PHONY: build clean ingest ingest-fast query test

GO_DIR    = go-ingestor
GO_BIN    = $(GO_DIR)/ingestor
CORPUS    ?= ./data/test_corpus

# Build Go ingestor binary
build:
	cd $(GO_DIR) && go build -o ingestor .

# Fast ingestion (Go parse/chunk + Python embed/index)
ingest-fast: build
	FAST_INGEST=1 python pipeline.py ingest $(CORPUS)

# Original Python-only ingestion (fallback)
ingest:
	python pipeline.py ingest $(CORPUS)

# Run fast_ingest.py standalone
ingest-standalone: build
	python fast_ingest.py $(CORPUS)

# Query phase
query:
	python pipeline.py query $(QUESTIONS)

# Clean build artifacts
clean:
	rm -f $(GO_BIN) data/chunks.arrow

# Run test pipeline
test: build
	python test_pipeline.py
