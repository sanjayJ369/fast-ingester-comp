package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/jsanjay/go-ingestor/chunker"
	"github.com/jsanjay/go-ingestor/embedder"
	"github.com/jsanjay/go-ingestor/parser"
	"github.com/jsanjay/go-ingestor/pipeline"
	"github.com/jsanjay/go-ingestor/writer"
)

const (
	chunkSize    = 512 // must match config.py CHUNK_SIZE
	chunkOverlap = 64  // must match config.py CHUNK_OVERLAP
	embedBatch   = 64  // ONNX batch size
)

func main() {
	embedFlag := flag.Bool("embed", false, "Run ONNX embedding in Go")
	modelDir := flag.String("model-dir", "./models/arctic-embed-xs-onnx", "Path to ONNX model directory")
	onnxLib := flag.String("onnx-lib", "/opt/homebrew/lib/libonnxruntime.dylib", "Path to ONNX Runtime shared library")
	flag.Parse()

	args := flag.Args()
	if len(args) < 2 {
		fmt.Fprintf(os.Stderr, "Usage: %s [--embed] [--model-dir DIR] <corpus_dir> <output.arrow>\n", os.Args[0])
		os.Exit(1)
	}

	corpusDir := args[0]
	outPath := args[1]
	numCPU := runtime.NumCPU()

	t0 := time.Now()

	// ── Stage 0: Discover files ─────────────────────────────────────────
	files, err := discoverFiles(corpusDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "[GO] Error discovering files: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("[GO] Found %d files in %s (%d CPUs, embed=%v)\n", len(files), corpusDir, numCPU, *embedFlag)

	// ── Pipeline channels ───────────────────────────────────────────────
	pages := make(chan pipeline.PageResult, numCPU*4)
	chunks := make(chan pipeline.RawChunk, numCPU*8)

	// ── Stage 1: PARSERS ────────────────────────────────────────────────
	var parseWg sync.WaitGroup
	tParse := time.Now()

	for _, f := range files {
		parseWg.Add(1)
		go parser.ParseToChannel(f, pages, &parseWg)
	}

	go func() {
		parseWg.Wait()
		close(pages)
		fmt.Printf("[GO] Parse stage done: %s\n", time.Since(tParse).Round(time.Millisecond))
	}()

	// ── Stage 2: CHUNKER POOL ───────────────────────────────────────────
	go chunker.RunChunkerPool(numCPU, chunkSize, chunkOverlap, pages, chunks)

	// ── Stage 3: COLLECTOR ──────────────────────────────────────────────
	var rawChunks []pipeline.RawChunk
	for rc := range chunks {
		rawChunks = append(rawChunks, rc)
	}

	tPipelineDone := time.Since(t0)
	fmt.Printf("[GO] Pipeline done: %d raw chunks in %s\n",
		len(rawChunks), tPipelineDone.Round(time.Millisecond))

	// ── Stage 4: Assign IDs ─────────────────────────────────────────────
	tAssign := time.Now()
	finalChunks := chunker.AssignChunkIDs(rawChunks)
	fmt.Printf("[GO] ID assignment: %d chunks in %s\n",
		len(finalChunks), time.Since(tAssign).Round(time.Millisecond))

	// ── Stage 5: Embed (optional) ───────────────────────────────────────
	if *embedFlag {
		tEmbed := time.Now()

		tokenizerPath := filepath.Join(*modelDir, "tokenizer.json")
		onnxModelPath := filepath.Join(*modelDir, "model.onnx")

		emb, err := embedder.New(onnxModelPath, tokenizerPath, *onnxLib)
		if err != nil {
			fmt.Fprintf(os.Stderr, "[GO] Error creating embedder: %v\n", err)
			os.Exit(1)
		}
		defer emb.Close()

		// Extract texts for embedding
		texts := make([]string, len(finalChunks))
		for i, c := range finalChunks {
			texts[i] = c.Text
		}

		fmt.Printf("[GO-EMBED] Embedding %d chunks (batch=%d)...\n", len(texts), embedBatch)
		embedResults, err := emb.EmbedBatchParallel(texts, embedBatch)
		if err != nil {
			fmt.Fprintf(os.Stderr, "[GO] Embedding error: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("[GO-EMBED] Done: %s\n", time.Since(tEmbed).Round(time.Millisecond))

		// Write Arrow with embeddings
		tArrow := time.Now()
		if err := writer.WriteArrowIPCWithEmbeddings(finalChunks, embedResults, outPath); err != nil {
			fmt.Fprintf(os.Stderr, "[GO] Error writing Arrow: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("[GO] Arrow write: %s\n", time.Since(tArrow).Round(time.Millisecond))
	} else {
		// Write Arrow without embeddings (Python will embed)
		tArrow := time.Now()
		if err := writer.WriteArrowIPC(finalChunks, outPath); err != nil {
			fmt.Fprintf(os.Stderr, "[GO] Error writing Arrow: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("[GO] Arrow write: %s\n", time.Since(tArrow).Round(time.Millisecond))
	}

	fmt.Printf("[GO] Total: %s\n", time.Since(t0).Round(time.Millisecond))
}

func discoverFiles(dir string) ([]string, error) {
	var files []string
	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			return nil
		}
		ext := strings.ToLower(filepath.Ext(path))
		if parser.SupportedExtensions[ext] {
			files = append(files, path)
		}
		return nil
	})
	return files, err
}
