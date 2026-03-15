package parser

import (
	"os"
	"fmt"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"

	"github.com/jsanjay/go-ingestor/pipeline"
)

const pagesPerWorker = 100 // each goroutine handles this many pages

// getPDFPageCount uses pdfinfo (poppler) to get total page count.
func getPDFPageCount(path string) (int, error) {
	cmd := exec.Command("pdfinfo", path)
	out, err := cmd.Output()
	if err != nil {
		return 0, fmt.Errorf("pdfinfo failed: %w", err)
	}
	for _, line := range strings.Split(string(out), "\n") {
		if strings.HasPrefix(line, "Pages:") {
			field := strings.TrimSpace(strings.TrimPrefix(line, "Pages:"))
			return strconv.Atoi(field)
		}
	}
	return 0, fmt.Errorf("pdfinfo: no page count found")
}

// pdfExtractRange runs pdftotext on a specific page range and returns pages.
// firstPage and lastPage are 1-indexed (pdftotext convention).
func pdfExtractRange(path string, firstPage, lastPage int) ([]string, error) {
	cmd := exec.Command("pdftotext",
		"-layout",
		"-f", strconv.Itoa(firstPage),
		"-l", strconv.Itoa(lastPage),
		path, "-",
	)
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("pdftotext pages %d-%d failed: %w", firstPage, lastPage, err)
	}

	rawPages := strings.Split(string(out), "\f")
	var pages []string
	for _, p := range rawPages {
		trimmed := strings.TrimSpace(p)
		if trimmed != "" {
			pages = append(pages, trimmed)
		}
	}
	return pages, nil
}

// ParsePDFToChannel splits a PDF across multiple goroutines (one per page range)
// and emits PageResult structs into the pages channel.
func ParsePDFToChannel(path string, pages chan<- pipeline.PageResult, wg *sync.WaitGroup) {
	defer wg.Done()

	base := filepath.Base(path)
	stem := strings.TrimSuffix(base, filepath.Ext(base))

	totalPages, err := getPDFPageCount(path)
	if err != nil {
		// Fallback: single pdftotext call for the whole file
		fmt.Fprintf(os.Stderr, "[GO-PDF] Warning: can't get page count for %s, using single pass: %v\n", base, err)
		allPages, err := pdfExtractRange(path, 1, 999999)
		if err != nil {
			fmt.Fprintf(os.Stderr, "[GO-PDF] Error parsing %s: %v\n", base, err)
			return
		}
		for i, text := range allPages {
			pages <- pipeline.PageResult{
				DocID:    stem,
				Filename: base,
				PageNum:  i + 1,
				Text:     text,
			}
		}
		return
	}

	fmt.Fprintf(os.Stderr, "[GO-PDF] %s: %d pages, splitting across %d workers\n",
		base, totalPages, (totalPages+pagesPerWorker-1)/pagesPerWorker)

	// Split into ranges and process each in its own goroutine
	var rangeWg sync.WaitGroup
	for start := 1; start <= totalPages; start += pagesPerWorker {
		end := start + pagesPerWorker - 1
		if end > totalPages {
			end = totalPages
		}

		rangeWg.Add(1)
		go func(first, last, pageOffset int) {
			defer rangeWg.Done()

			extracted, err := pdfExtractRange(path, first, last)
			if err != nil {
				fmt.Fprintf(os.Stderr, "[GO-PDF] Error on %s pages %d-%d: %v\n", base, first, last, err)
				return
			}

			for i, text := range extracted {
				pages <- pipeline.PageResult{
					DocID:    stem,
					Filename: base,
					PageNum:  pageOffset + i + 1, // global 1-indexed page number
					Text:     text,
				}
			}
		}(start, end, start-1)
	}

	rangeWg.Wait()
}

// parsePDF is the legacy single-pass PDF parser (kept for Parse() compat).
func parsePDF(path string) (ParsedDocument, error) {
	cmd := exec.Command("pdftotext", "-layout", path, "-")
	out, err := cmd.Output()
	if err != nil {
		return ParsedDocument{}, fmt.Errorf("pdftotext failed: %w", err)
	}

	base := filepath.Base(path)
	stem := strings.TrimSuffix(base, filepath.Ext(base))

	rawPages := strings.Split(string(out), "\f")
	var pgs []string
	for _, p := range rawPages {
		trimmed := strings.TrimSpace(p)
		if trimmed != "" {
			pgs = append(pgs, trimmed)
		}
	}

	return ParsedDocument{
		DocID:    stem,
		Filename: base,
		Pages:    pgs,
	}, nil
}
