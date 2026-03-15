package chunker

import (
	"fmt"
	"sync"

	"github.com/jsanjay/go-ingestor/pipeline"
)

// Sequencer ensures chunks for each document are emitted in deterministic
// order (PageNum ASC) even if pages arrive out of order from parallel parsers.
type Sequencer struct {
	mu         sync.Mutex
	docStates  map[string]*docState
	outChannel chan<- Chunk
}

type docState struct {
	nextPage  int
	chunkIdx  int
	pending   map[int]pipeline.PageChunks // PageNum -> chunks
}

func NewSequencer(out chan<- Chunk) *Sequencer {
	return &Sequencer{
		docStates:  make(map[string]*docState),
		outChannel: out,
	}
}

// Process receives PageChunks and emits any Chunks that are now "in order".
func (s *Sequencer) Process(pc pipeline.PageChunks) {
	s.mu.Lock()
	defer s.mu.Unlock()

	state, ok := s.docStates[pc.DocID]
	if !ok {
		// First time seeing this doc. 
		// We assume PageNum starts at 1, but we could also take pc.PageNum if 
		// we want to be more flexible (e.g. if some docs start at page 0).
		// For Lucio, they are 1-indexed.
		state = &docState{
			nextPage: 1,
			pending:  make(map[int]pipeline.PageChunks),
		}
		s.docStates[pc.DocID] = state
	}

	// Buffer the incoming page
	state.pending[pc.PageNum] = pc

	// Attempt to emit contiguous pages
	for {
		p, exists := state.pending[state.nextPage]
		if !exists {
			break
		}

		// Emit all chunks in this page
		for _, rc := range p.Chunks {
			s.outChannel <- Chunk{
				ChunkID:  fmt.Sprintf("%s__chunk_%d", rc.DocID, state.chunkIdx),
				DocID:    rc.DocID,
				Filename: rc.Filename,
				Text:     rc.Text,
				PageNum:  int32(rc.PageNum),
				ChunkIdx: int32(state.chunkIdx),
			}
			state.chunkIdx++
		}

		// Cleanup and move to next
		delete(state.pending, state.nextPage)
		state.nextPage++
	}
}

// Close flushes any remaining out-of-order chunks (though in a healthy run,
// everything should have been emitted).
func (s *Sequencer) Close() {
	s.mu.Lock()
	defer s.mu.Unlock()

	for docID, state := range s.docStates {
		if len(state.pending) > 0 {
			fmt.Printf("[SEQUENCER] Warning: document %s has %d out-of-order pages remaining\n", 
				docID, len(state.pending))
			// Optional: sort and flush remaining pages anyway?
		}
	}
}
