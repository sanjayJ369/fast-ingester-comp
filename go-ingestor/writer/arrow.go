package writer

import (
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/ipc"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/jsanjay/go-ingestor/chunker"
	"github.com/jsanjay/go-ingestor/embedder"
)

// sanitizeUTF8 replaces invalid UTF-8 bytes with the Unicode replacement character.
func sanitizeUTF8(s string) string {
	return strings.ToValidUTF8(s, "\uFFFD")
}

// SimpleSchema includes chunk metadata without embeddings.
var SimpleSchema = arrow.NewSchema([]arrow.Field{
	{Name: "chunk_id", Type: arrow.BinaryTypes.String},
	{Name: "doc_id", Type: arrow.BinaryTypes.String},
	{Name: "filename", Type: arrow.BinaryTypes.String},
	{Name: "text", Type: arrow.BinaryTypes.String},
	{Name: "page_num", Type: arrow.PrimitiveTypes.Int32},
	{Name: "chunk_idx", Type: arrow.PrimitiveTypes.Int32},
}, nil)

// SchemaWithEmbeddings includes chunk metadata + embedding vectors.
var SchemaWithEmbeddings = arrow.NewSchema([]arrow.Field{
	{Name: "chunk_id", Type: arrow.BinaryTypes.String},
	{Name: "doc_id", Type: arrow.BinaryTypes.String},
	{Name: "filename", Type: arrow.BinaryTypes.String},
	{Name: "text", Type: arrow.BinaryTypes.String},
	{Name: "page_num", Type: arrow.PrimitiveTypes.Int32},
	{Name: "chunk_idx", Type: arrow.PrimitiveTypes.Int32},
	{Name: "full_vec", Type: arrow.ListOf(arrow.PrimitiveTypes.Float32)},
	{Name: "coarse_vec", Type: arrow.ListOf(arrow.PrimitiveTypes.Float32)},
}, nil)

// StreamArrowIPC streams chunks from a channel to an io.Writer using Arrow IPC streaming format.
func StreamArrowIPC(chunks <-chan chunker.Chunk, out io.Writer) error {
	pool := memory.NewGoAllocator()
	w := ipc.NewWriter(out, ipc.WithSchema(SimpleSchema))
	defer w.Close()

	const batchSize = 100
	const flushInterval = 500 * time.Millisecond

	b := array.NewRecordBuilder(pool, SimpleSchema)
	defer b.Release()

	count := 0
	lastFlush := time.Now()

	flush := func() error {
		if count == 0 {
			return nil
		}
		rec := b.NewRecord()
		defer rec.Release()
		if err := w.Write(rec); err != nil {
			return err
		}
		count = 0
		lastFlush = time.Now()
		return nil
	}

	for {
		select {
		case c, ok := <-chunks:
			if !ok {
				return flush()
			}
			b.Field(0).(*array.StringBuilder).Append(c.ChunkID)
			b.Field(1).(*array.StringBuilder).Append(c.DocID)
			b.Field(2).(*array.StringBuilder).Append(c.Filename)
			b.Field(3).(*array.StringBuilder).Append(sanitizeUTF8(c.Text))
			b.Field(4).(*array.Int32Builder).Append(c.PageNum)
			b.Field(5).(*array.Int32Builder).Append(c.ChunkIdx)
			count++

			if count >= batchSize {
				if err := flush(); err != nil {
					return err
				}
			}
		case <-time.After(flushInterval):
			if time.Since(lastFlush) >= flushInterval {
				if err := flush(); err != nil {
					return err
				}
			}
		}
	}
}

// WriteArrowIPCWithEmbeddings writes chunks + embeddings to an Arrow IPC file.
func WriteArrowIPCWithEmbeddings(chunks []chunker.Chunk, embeddings []embedder.EmbedResult, outPath string) error {
	pool := memory.NewGoAllocator()
	b := array.NewRecordBuilder(pool, SchemaWithEmbeddings)
	defer b.Release()

	for i, c := range chunks {
		b.Field(0).(*array.StringBuilder).Append(c.ChunkID)
		b.Field(1).(*array.StringBuilder).Append(c.DocID)
		b.Field(2).(*array.StringBuilder).Append(c.Filename)
		b.Field(3).(*array.StringBuilder).Append(sanitizeUTF8(c.Text))
		b.Field(4).(*array.Int32Builder).Append(c.PageNum)
		b.Field(5).(*array.Int32Builder).Append(c.ChunkIdx)

		// Full vector (384-dim)
		fullListBuilder := b.Field(6).(*array.ListBuilder)
		fullValBuilder := fullListBuilder.ValueBuilder().(*array.Float32Builder)
		fullListBuilder.Append(true)
		fullValBuilder.AppendValues(embeddings[i].FullVec, nil)

		// Coarse vector (128-dim)
		coarseListBuilder := b.Field(7).(*array.ListBuilder)
		coarseValBuilder := coarseListBuilder.ValueBuilder().(*array.Float32Builder)
		coarseListBuilder.Append(true)
		coarseValBuilder.AppendValues(embeddings[i].CoarseVec, nil)
	}

	rec := b.NewRecord()
	defer rec.Release()

	f, err := os.Create(outPath)
	if err != nil {
		return fmt.Errorf("create %s: %w", outPath, err)
	}
	defer f.Close()

	w, err := ipc.NewFileWriter(f, ipc.WithSchema(SchemaWithEmbeddings))
	if err != nil {
		return fmt.Errorf("create arrow writer: %w", err)
	}

	if err := w.Write(rec); err != nil {
		return fmt.Errorf("write record: %w", err)
	}

	if err := w.Close(); err != nil {
		return fmt.Errorf("close writer: %w", err)
	}

	fmt.Printf("[GO-ARROW] Wrote %d chunks (with embeddings) to %s\n", len(chunks), outPath)
	return nil
}

// WriteArrowIPC writes chunks without embeddings (legacy).
func WriteArrowIPC(chunks []chunker.Chunk, outPath string) error {
	pool := memory.NewGoAllocator()
	b := array.NewRecordBuilder(pool, SimpleSchema)
	defer b.Release()

	for _, c := range chunks {
		b.Field(0).(*array.StringBuilder).Append(c.ChunkID)
		b.Field(1).(*array.StringBuilder).Append(c.DocID)
		b.Field(2).(*array.StringBuilder).Append(c.Filename)
		b.Field(3).(*array.StringBuilder).Append(sanitizeUTF8(c.Text))
		b.Field(4).(*array.Int32Builder).Append(c.PageNum)
		b.Field(5).(*array.Int32Builder).Append(c.ChunkIdx)
	}

	rec := b.NewRecord()
	defer rec.Release()

	f, err := os.Create(outPath)
	if err != nil {
		return fmt.Errorf("create %s: %w", outPath, err)
	}
	defer f.Close()

	w, err := ipc.NewFileWriter(f, ipc.WithSchema(SimpleSchema))
	if err != nil {
		return fmt.Errorf("create arrow writer: %w", err)
	}
	if err := w.Write(rec); err != nil {
		return fmt.Errorf("write record: %w", err)
	}
	if err := w.Close(); err != nil {
		return fmt.Errorf("close writer: %w", err)
	}

	fmt.Printf("[GO-ARROW] Wrote %d chunks to %s\n", len(chunks), outPath)
	return nil
}
