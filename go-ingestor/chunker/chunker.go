package chunker

import (
	"fmt"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/jsanjay/go-ingestor/pipeline"
)

// Chunk is the final output with assigned IDs, ready for Arrow.
type Chunk struct {
	ChunkID  string
	DocID    string
	Filename string
	Text     string
	PageNum  int32
	ChunkIdx int32
}

// RunChunkerPool starts numWorkers goroutines that consume PageResults from
// the pages channel, chunk each page, and emit Chunks into the chunks channel.
// Closes the chunks channel when all workers are done.
func RunChunkerPool(numWorkers, size, overlap int, pages <-chan pipeline.PageResult, chunks chan<- Chunk) {
	var wg sync.WaitGroup
	var globalIdx int32

	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for page := range pages {
				if strings.TrimSpace(page.Text) == "" {
					continue
				}
				splits := splitText(page.Text, size, overlap)
				for _, s := range splits {
					trimmed := strings.TrimSpace(s)
					if trimmed == "" {
						continue
					}

					idx := atomic.AddInt32(&globalIdx, 1) - 1
					chunks <- Chunk{
						ChunkID:  fmt.Sprintf("%s__chunk_%d", page.DocID, idx),
						DocID:    page.DocID,
						Filename: page.Filename,
						Text:     trimmed,
						PageNum:  int32(page.PageNum),
						ChunkIdx: idx,
					}
				}
			}
		}()
	}
	wg.Wait()
	close(chunks)
}

// splitText is an exact port of _split_text from ingestion/chunker.py:29-53.
func splitText(text string, size, overlap int) []string {
	if len(text) <= size {
		return []string{text}
	}

	separators := []string{"\n\n", "\n", ". ", " "}
	for _, sep := range separators {
		idx := strings.LastIndex(text[:size], sep)
		if idx > size/2 {
			splitAt := idx + len(sep)
			head := strings.TrimSpace(text[:splitAt])
			tailStart := splitAt - overlap
			if tailStart < 0 {
				tailStart = 0
			}
			tail := strings.TrimSpace(text[tailStart:])
			return append([]string{head}, splitText(tail, size, overlap)...)
		}
	}

	head := text[:size]
	tailStart := size - overlap
	if tailStart < 0 {
		tailStart = 0
	}
	tail := text[tailStart:]
	return append([]string{head}, splitText(tail, size, overlap)...)
}

// ChunkDocument is the legacy single-doc API (kept for backward compat).
func ChunkDocument(doc struct {
	DocID    string
	Filename string
	Pages    []string
}, size, overlap int) []Chunk {
	var chunks []Chunk
	globalIdx := 0

	for pageNum, pageText := range doc.Pages {
		if strings.TrimSpace(pageText) == "" {
			continue
		}
		splits := splitText(pageText, size, overlap)
		for _, s := range splits {
			trimmed := strings.TrimSpace(s)
			if trimmed == "" {
				continue
			}
			chunks = append(chunks, Chunk{
				ChunkID:  fmt.Sprintf("%s__chunk_%d", doc.DocID, globalIdx),
				DocID:    doc.DocID,
				Filename: doc.Filename,
				Text:     trimmed,
				PageNum:  int32(pageNum + 1),
				ChunkIdx: int32(globalIdx),
			})
			globalIdx++
		}
	}
	return chunks
}
