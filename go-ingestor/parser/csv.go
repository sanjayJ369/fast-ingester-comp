package parser

import (
	"encoding/csv"
	"os"
	"path/filepath"
	"strings"
)

func parseCSV(path string) (ParsedDocument, error) {
	f, err := os.Open(path)
	if err != nil {
		return ParsedDocument{}, err
	}
	defer f.Close()

	base := filepath.Base(path)
	stem := strings.TrimSuffix(base, filepath.Ext(base))

	reader := csv.NewReader(f)
	reader.LazyQuotes = true
	reader.FieldsPerRecord = -1 // allow variable fields

	records, err := reader.ReadAll()
	if err != nil {
		return ParsedDocument{}, err
	}

	// Convert rows to tab-separated strings
	var rows []string
	for _, record := range records {
		rows = append(rows, strings.Join(record, "\t"))
	}

	// Chunk into 50-row pages (matching Python behavior)
	var pages []string
	for i := 0; i < len(rows); i += 50 {
		end := i + 50
		if end > len(rows) {
			end = len(rows)
		}
		pages = append(pages, strings.Join(rows[i:end], "\n"))
	}

	return ParsedDocument{
		DocID:    stem,
		Filename: base,
		Pages:    pages,
	}, nil
}
