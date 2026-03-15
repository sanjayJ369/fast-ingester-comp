package chunker

import (
	"fmt"
	"testing"

	"github.com/jsanjay/go-ingestor/pipeline"
)

func TestSequencer(t *testing.T) {
	out := make(chan Chunk, 100)
	s := NewSequencer(out)

	// doc1: page 2 arrives before page 1
	p2 := pipeline.PageChunks{
		DocID:    "doc1",
		Filename: "file1.pdf",
		PageNum:  2,
		Chunks: []pipeline.RawChunk{
			{DocID: "doc1", PageNum: 2, Text: "Page 2 - Chunk 0", PageOrder: 0},
		},
	}
	p1 := pipeline.PageChunks{
		DocID:    "doc1",
		Filename: "file1.pdf",
		PageNum:  1,
		Chunks: []pipeline.RawChunk{
			{DocID: "doc1", PageNum: 1, Text: "Page 1 - Chunk 0", PageOrder: 0},
			{DocID: "doc1", PageNum: 1, Text: "Page 1 - Chunk 1", PageOrder: 1},
		},
	}

	// Process page 2 first
	s.Process(p2)
	select {
	case <-out:
		t.Fatal("Should not have emitted anything yet")
	default:
	}

	// Process page 1
	s.Process(p1)

	// Now we should get 3 chunks in order
	expected := []struct {
		pageNum  int32
		chunkIdx int32
		text     string
	}{
		{1, 0, "Page 1 - Chunk 0"},
		{1, 1, "Page 1 - Chunk 1"},
		{2, 2, "Page 2 - Chunk 0"},
	}

	for i, exp := range expected {
		select {
		case c := <-out:
			if c.PageNum != exp.pageNum || c.ChunkIdx != exp.chunkIdx || c.Text != exp.text {
				t.Errorf("Chunk %d: expected %+v, got %+v", i, exp, c)
			}
			expectedChunkID := fmt.Sprintf("doc1__chunk_%d", exp.chunkIdx)
			if c.ChunkID != expectedChunkID {
				t.Errorf("Chunk %d ID: expected %s, got %s", i, expectedChunkID, c.ChunkID)
			}
		default:
			t.Fatalf("Missing chunk %d", i)
		}
	}
}
