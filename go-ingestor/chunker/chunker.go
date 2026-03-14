package chunker

import (
	"fmt"
	"strings"
	"sync"

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
// the pages channel, chunk each page, and emit RawChunks into the chunks channel.
// Closes the chunks channel when all workers are done.
func RunChunkerPool(numWorkers, size, overlap int, pages <-chan pipeline.PageResult, chunks chan<- pipeline.RawChunk) {
	var wg sync.WaitGroup
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for page := range pages {
				if strings.TrimSpace(page.Text) == "" {
					continue
				}
				splits := splitText(page.Text, size, overlap)
				for order, s := range splits {
					trimmed := strings.TrimSpace(s)
					if trimmed == "" {
						continue
					}
					chunks <- pipeline.RawChunk{
						DocID:     page.DocID,
						Filename:  page.Filename,
						PageNum:   page.PageNum,
						Text:      trimmed,
						PageOrder: order,
					}
				}
			}
		}()
	}
	wg.Wait()
	close(chunks)
}

// AssignChunkIDs takes collected raw chunks, sorts by (doc_id, page_num, page_order),
// and assigns sequential chunk_idx per document. Returns final Chunk slice.
func AssignChunkIDs(raw []pipeline.RawChunk) []Chunk {
	// Group by doc_id, preserving page ordering
	type docChunks struct {
		chunks []pipeline.RawChunk
	}
	docs := make(map[string]*docChunks)
	docOrder := []string{} // preserve insertion order

	for _, r := range raw {
		dc, ok := docs[r.DocID]
		if !ok {
			dc = &docChunks{}
			docs[r.DocID] = dc
			docOrder = append(docOrder, r.DocID)
		}
		dc.chunks = append(dc.chunks, r)
	}

	// Sort each doc's chunks by (page_num, page_order) and assign IDs
	var result []Chunk
	for _, docID := range docOrder {
		dc := docs[docID]
		sortRawChunks(dc.chunks)

		for i, r := range dc.chunks {
			result = append(result, Chunk{
				ChunkID:  fmt.Sprintf("%s__chunk_%d", r.DocID, i),
				DocID:    r.DocID,
				Filename: r.Filename,
				Text:     r.Text,
				PageNum:  int32(r.PageNum),
				ChunkIdx: int32(i),
			})
		}
	}

	return result
}

// sortRawChunks sorts by (PageNum, PageOrder) using a simple insertion sort
// (stable, good for mostly-ordered data coming from channels).
func sortRawChunks(chunks []pipeline.RawChunk) {
	for i := 1; i < len(chunks); i++ {
		key := chunks[i]
		j := i - 1
		for j >= 0 && less(key, chunks[j]) {
			chunks[j+1] = chunks[j]
			j--
		}
		chunks[j+1] = key
	}
}

func less(a, b pipeline.RawChunk) bool {
	if a.PageNum != b.PageNum {
		return a.PageNum < b.PageNum
	}
	return a.PageOrder < b.PageOrder
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
