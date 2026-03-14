# Stage 1: Build Go ingestor
FROM golang:1.22-alpine AS go-builder
WORKDIR /build
COPY go-ingestor/ .
RUN go build -o /ingestor .

# Stage 2: Python runtime
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libffi-dev poppler-utils && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt einops

# Pre-download embedding model at build time
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)"

# Copy Go binary from builder stage
COPY --from=go-builder /ingestor /app/go-ingestor/ingestor

COPY . .

EXPOSE 8000
ENV FAST_INGEST=1
CMD ["python", "pipeline.py", "ingest", "./data/corpus"]
