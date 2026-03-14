package embedder

import (
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/daulet/tokenizers"
	ort "github.com/yalue/onnxruntime_go"
)

const (
	EmbedDimFull   = 384
	EmbedDimCoarse = 128
	MaxSeqLen      = 512
)

// Embedder wraps the ONNX model and tokenizer for generating embeddings.
type Embedder struct {
	tokenizer *tokenizers.Tokenizer
	modelPath string
	mu        sync.Mutex

	// Pre-allocated session (created lazily on first batch)
	session           *ort.AdvancedSession
	sessionBatchSize  int
	inputIDsTensor    *ort.Tensor[int64]
	attentionTensor   *ort.Tensor[int64]
	typeIDsTensor     *ort.Tensor[int64]
	outputTensor      *ort.Tensor[float32]
	inputIDsData      []int64
	attentionData     []int64
	typeIDsData       []int64
	outputData        []float32
}

// EmbedResult holds the embedding vectors for a single chunk.
type EmbedResult struct {
	FullVec   []float32 // 384-dim
	CoarseVec []float32 // 128-dim (truncated + re-normalized)
}

// New creates an Embedder. Call Close() when done.
func New(onnxModelPath, tokenizerPath, onnxLibPath string) (*Embedder, error) {
	ort.SetSharedLibraryPath(onnxLibPath)
	if err := ort.InitializeEnvironment(); err != nil {
		return nil, fmt.Errorf("init onnxruntime: %w", err)
	}

	tk, err := tokenizers.FromFile(tokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("load tokenizer: %w", err)
	}

	return &Embedder{
		tokenizer: tk,
		modelPath: onnxModelPath,
	}, nil
}

// initSession creates the ONNX session with fixed tensor shapes for reuse.
func (e *Embedder) initSession(batchSize int) error {
	inputShape := ort.Shape{int64(batchSize), int64(MaxSeqLen)}
	outputShape := ort.Shape{int64(batchSize), int64(MaxSeqLen), int64(EmbedDimFull)}

	e.inputIDsData = make([]int64, batchSize*MaxSeqLen)
	e.attentionData = make([]int64, batchSize*MaxSeqLen)
	e.typeIDsData = make([]int64, batchSize*MaxSeqLen)
	e.outputData = make([]float32, batchSize*MaxSeqLen*EmbedDimFull)

	var err error
	e.inputIDsTensor, err = ort.NewTensor(inputShape, e.inputIDsData)
	if err != nil {
		return fmt.Errorf("create input_ids tensor: %w", err)
	}

	e.attentionTensor, err = ort.NewTensor(inputShape, e.attentionData)
	if err != nil {
		return fmt.Errorf("create attention_mask tensor: %w", err)
	}

	e.typeIDsTensor, err = ort.NewTensor(inputShape, e.typeIDsData)
	if err != nil {
		return fmt.Errorf("create token_type_ids tensor: %w", err)
	}

	e.outputTensor, err = ort.NewTensor(outputShape, e.outputData)
	if err != nil {
		return fmt.Errorf("create output tensor: %w", err)
	}

	e.session, err = ort.NewAdvancedSession(
		e.modelPath,
		[]string{"input_ids", "attention_mask", "token_type_ids"},
		[]string{"last_hidden_state"},
		[]ort.ArbitraryTensor{e.inputIDsTensor, e.attentionTensor, e.typeIDsTensor},
		[]ort.ArbitraryTensor{e.outputTensor},
		nil,
	)
	if err != nil {
		return fmt.Errorf("create session: %w", err)
	}

	e.sessionBatchSize = batchSize
	return nil
}

// Close releases resources.
func (e *Embedder) Close() {
	if e.session != nil {
		e.session.Destroy()
	}
	if e.inputIDsTensor != nil {
		e.inputIDsTensor.Destroy()
	}
	if e.attentionTensor != nil {
		e.attentionTensor.Destroy()
	}
	if e.typeIDsTensor != nil {
		e.typeIDsTensor.Destroy()
	}
	if e.outputTensor != nil {
		e.outputTensor.Destroy()
	}
	if e.tokenizer != nil {
		e.tokenizer.Close()
	}
	ort.DestroyEnvironment()
}

// EmbedBatchParallel embeds texts using a pre-allocated ONNX session.
func (e *Embedder) EmbedBatchParallel(texts []string, batchSize int) ([]EmbedResult, error) {
	// Initialize session on first call
	if e.session == nil {
		fmt.Printf("[GO-EMBED] Initializing ONNX session (batch=%d, seq=%d)...\n", batchSize, MaxSeqLen)
		t0 := time.Now()
		if err := e.initSession(batchSize); err != nil {
			return nil, err
		}
		fmt.Printf("[GO-EMBED] Session ready in %s\n", time.Since(t0).Round(time.Millisecond))
	}

	results := make([]EmbedResult, len(texts))
	t0 := time.Now()
	totalBatches := (len(texts) + batchSize - 1) / batchSize

	for i := 0; i < len(texts); i += batchSize {
		end := i + batchSize
		if end > len(texts) {
			end = len(texts)
		}

		batchResults, err := e.embedSingleBatch(texts[i:end])
		if err != nil {
			return nil, fmt.Errorf("batch %d-%d: %w", i, end, err)
		}
		copy(results[i:end], batchResults)

		batchNum := i/batchSize + 1
		if batchNum%10 == 0 || batchNum == totalBatches {
			fmt.Printf("[GO-EMBED] ... %d/%d (batch %d/%d, %s)\n",
				end, len(texts), batchNum, totalBatches,
				time.Since(t0).Round(time.Millisecond))
		}
	}

	return results, nil
}

// embedSingleBatch processes a single batch through the reusable ONNX session.
func (e *Embedder) embedSingleBatch(texts []string) ([]EmbedResult, error) {
	actualBatch := len(texts)

	// 1. Tokenize all texts
	seqLens := make([]int, e.sessionBatchSize)

	// Zero out input buffers
	for j := range e.inputIDsData {
		e.inputIDsData[j] = 0
		e.attentionData[j] = 0
		e.typeIDsData[j] = 0
	}

	for i, text := range texts {
		ids, _ := e.tokenizer.Encode(text, true)

		// Truncate to MaxSeqLen
		seqLen := len(ids)
		if seqLen > MaxSeqLen {
			seqLen = MaxSeqLen
		}
		seqLens[i] = seqLen

		// Fill padded input at fixed positions (always MaxSeqLen stride)
		base := i * MaxSeqLen
		for j := 0; j < seqLen; j++ {
			e.inputIDsData[base+j] = int64(ids[j])
			e.attentionData[base+j] = 1
			// typeIDsData stays 0
		}
	}

	// Pad remaining batch slots with zeros (already zeroed above)
	for i := actualBatch; i < e.sessionBatchSize; i++ {
		seqLens[i] = 1 // avoid div-by-zero in mean pool (won't be used)
		// Set at least one attention token so ONNX doesn't get confused
		e.attentionData[i*MaxSeqLen] = 1
	}

	// 2. Run ONNX session (reuses pre-allocated tensors)
	if err := e.session.Run(); err != nil {
		return nil, fmt.Errorf("run session: %w", err)
	}

	// 3. Mean pooling + normalization (only for actual batch items)
	results := make([]EmbedResult, actualBatch)

	for i := 0; i < actualBatch; i++ {
		seqLen := seqLens[i]

		// Mean pool over non-padded tokens
		fullVec := make([]float32, EmbedDimFull)
		for t := 0; t < seqLen; t++ {
			offset := (i*MaxSeqLen + t) * EmbedDimFull
			for d := 0; d < EmbedDimFull; d++ {
				fullVec[d] += e.outputData[offset+d]
			}
		}
		invSeqLen := float32(1.0 / float64(seqLen))
		for d := 0; d < EmbedDimFull; d++ {
			fullVec[d] *= invSeqLen
		}

		// L2 normalize full vector
		l2normalize(fullVec)

		// Truncate to coarse + re-normalize
		coarseVec := make([]float32, EmbedDimCoarse)
		copy(coarseVec, fullVec[:EmbedDimCoarse])
		l2normalize(coarseVec)

		results[i] = EmbedResult{
			FullVec:   fullVec,
			CoarseVec: coarseVec,
		}
	}

	return results, nil
}

func l2normalize(vec []float32) {
	var norm float64
	for _, v := range vec {
		norm += float64(v) * float64(v)
	}
	norm = math.Sqrt(norm)
	if norm < 1e-9 {
		return
	}
	for i := range vec {
		vec[i] = float32(float64(vec[i]) / norm)
	}
}
