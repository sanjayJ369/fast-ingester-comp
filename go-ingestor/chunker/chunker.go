package chunker

import (
	"fmt"
	"sort"
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
// the pages channel, chunk each page, and emit PageChunks into the channel.
// Closes the chunks channel when all workers are done.
func RunChunkerPool(numWorkers, size, overlap int, pages <-chan pipeline.PageResult, chunks chan<- pipeline.PageChunks) {
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
				var pageChunks []pipeline.RawChunk
				for order, s := range splits {
					trimmed := strings.TrimSpace(s)
					if trimmed == "" {
						continue
					}
					pageChunks = append(pageChunks, pipeline.RawChunk{
						DocID:     page.DocID,
						Filename:  page.Filename,
						PageNum:   page.PageNum,
						Text:      trimmed,
						PageOrder: order,
					})
				}
				if len(pageChunks) > 0 {
					chunks <- pipeline.PageChunks{
						DocID:    page.DocID,
						Filename: page.Filename,
						PageNum:  page.PageNum,
						Chunks:   pageChunks,
					}
				}
			}
		}()
	}
	wg.Wait()
	close(chunks)
}

// AssignChunkIDs takes collected page chunks, sorts by (doc_id, page_num),
// and assigns sequential chunk_idx per document. Returns final Chunk slice.
func AssignChunkIDs(rawPages []pipeline.PageChunks) []Chunk {
	// Group by doc_id
	docs := make(map[string][]pipeline.PageChunks)
	var docOrder []string

	for _, rp := range rawPages {
		if _, ok := docs[rp.DocID]; !ok {
			docOrder = append(docOrder, rp.DocID)
		}
		docs[rp.DocID] = append(docs[rp.DocID], rp)
	}

	var result []Chunk
	for _, docID := range docOrder {
		pages := docs[docID]
		// Sort pages by PageNum
		sort.Slice(pages, func(i, j int) bool {
			return pages[i].PageNum < pages[j].PageNum
		})

		chunkIdx := 0
		for _, p := range pages {
			for _, rc := range p.Chunks {
				result = append(result, Chunk{
					ChunkID:  fmt.Sprintf("%s__chunk_%d", rc.DocID, chunkIdx),
					DocID:    rc.DocID,
					Filename: rc.Filename,
					Text:     rc.Text,
					PageNum:  int32(rc.PageNum),
					ChunkIdx: int32(chunkIdx),
				})
				chunkIdx++
			}
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
