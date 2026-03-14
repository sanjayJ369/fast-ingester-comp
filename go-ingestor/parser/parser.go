package parser

import (
	"fmt"
	"path/filepath"
	"strings"
	"sync"

	"github.com/jsanjay/go-ingestor/pipeline"
)

// ParsedDocument holds the text extracted from a single file.
type ParsedDocument struct {
	DocID    string
	Filename string
	Pages    []string // one entry per page or logical section
}

// SupportedExtensions lists all file types we can parse.
var SupportedExtensions = map[string]bool{
	".pdf":  true,
	".docx": true,
	".doc":  true,
	".txt":  true,
	".html": true,
	".htm":  true,
	".csv":  true,
	".xlsx": true,
	".xls":  true,
}

// ParseToChannel dispatches file parsing and emits pages into the channel.
// For PDFs, this spawns multiple goroutines per file (one per page range).
// For other formats, parses in one goroutine and emits all pages.
func ParseToChannel(path string, pages chan<- pipeline.PageResult, wg *sync.WaitGroup) {
	ext := strings.ToLower(filepath.Ext(path))

	if ext == ".pdf" {
		// PDF gets multi-threaded page-range splitting
		ParsePDFToChannel(path, pages, wg)
		return
	}

	// All other formats: parse normally, emit pages
	defer wg.Done()

	doc, err := parseNonPDF(path, ext)
	if err != nil {
		fmt.Printf("[GO] Warning: %v\n", err)
		return
	}

	for i, text := range doc.Pages {
		pages <- pipeline.PageResult{
			DocID:    doc.DocID,
			Filename: doc.Filename,
			PageNum:  i + 1,
			Text:     text,
		}
	}
}

// parseNonPDF handles all non-PDF formats.
func parseNonPDF(path, ext string) (ParsedDocument, error) {
	base := filepath.Base(path)
	stem := strings.TrimSuffix(base, filepath.Ext(base))

	var doc ParsedDocument
	var err error

	switch ext {
	case ".docx", ".doc":
		doc, err = parseDOCX(path)
	case ".html", ".htm":
		doc, err = parseHTML(path)
	case ".csv":
		doc, err = parseCSV(path)
	case ".xlsx", ".xls":
		doc, err = parseExcel(path)
	default:
		doc, err = parseTXT(path)
	}

	if err != nil {
		return ParsedDocument{}, fmt.Errorf("parse %s: %w", base, err)
	}

	if doc.DocID == "" {
		doc.DocID = stem
	}
	if doc.Filename == "" {
		doc.Filename = base
	}

	return doc, nil
}

// Parse is the legacy single-file API (kept for backward compat).
func Parse(path string) (ParsedDocument, error) {
	ext := strings.ToLower(filepath.Ext(path))
	base := filepath.Base(path)
	stem := strings.TrimSuffix(base, filepath.Ext(base))

	var doc ParsedDocument
	var err error

	switch ext {
	case ".pdf":
		doc, err = parsePDF(path)
	case ".docx", ".doc":
		doc, err = parseDOCX(path)
	case ".html", ".htm":
		doc, err = parseHTML(path)
	case ".csv":
		doc, err = parseCSV(path)
	case ".xlsx", ".xls":
		doc, err = parseExcel(path)
	default:
		doc, err = parseTXT(path)
	}

	if err != nil {
		return ParsedDocument{}, fmt.Errorf("parse %s: %w", base, err)
	}

	if doc.DocID == "" {
		doc.DocID = stem
	}
	if doc.Filename == "" {
		doc.Filename = base
	}

	return doc, nil
}
