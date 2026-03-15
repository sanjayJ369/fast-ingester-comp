package pipeline

// PageResult is emitted by parsers into the pages channel.
// Each represents one page/section of text from a document.
type PageResult struct {
	DocID    string
	Filename string
	PageNum  int // 1-indexed, preserves ordering for chunk ID assignment
	Text     string
}

// RawChunk is emitted by chunker workers.
// Has no final chunk_id yet — IDs are assigned by the collector after sorting.
type RawChunk struct {
	DocID     string
	Filename  string
	PageNum   int // from the source page
	Text      string
	PageOrder int // ordering within this page's chunks (0, 1, 2, ...)
}

// PageChunks is a set of all chunks belonging to a single page.
// Emitting this ensures the sequencer gets an atomic page.
type PageChunks struct {
	DocID    string
	Filename string
	PageNum  int
	Chunks   []RawChunk
}
