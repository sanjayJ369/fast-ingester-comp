package parser

import (
	"os"
	"path/filepath"
	"strings"
)

func parseTXT(path string) (ParsedDocument, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return ParsedDocument{}, err
	}

	text := string(data)
	base := filepath.Base(path)
	stem := strings.TrimSuffix(base, filepath.Ext(base))

	// Split on double newlines as paragraph boundaries
	paragraphs := splitNonEmpty(text, "\n\n")

	// Group into ~500 char pages (matching Python behavior)
	var pages []string
	var current []string
	count := 0

	for _, para := range paragraphs {
		para = strings.TrimSpace(para)
		if para == "" {
			continue
		}
		current = append(current, para)
		count += len(para)
		if count >= 500 {
			pages = append(pages, strings.Join(current, "\n\n"))
			current = nil
			count = 0
		}
	}
	if len(current) > 0 {
		pages = append(pages, strings.Join(current, "\n\n"))
	}

	// Fallback: if no pages, use first 2000 chars
	if len(pages) == 0 {
		fallback := text
		if len(fallback) > 2000 {
			fallback = fallback[:2000]
		}
		pages = []string{fallback}
	}

	return ParsedDocument{
		DocID:    stem,
		Filename: base,
		Pages:    pages,
	}, nil
}

// splitNonEmpty splits text by separator and returns non-empty trimmed parts.
func splitNonEmpty(text, sep string) []string {
	parts := strings.Split(text, sep)
	var result []string
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			result = append(result, p)
		}
	}
	return result
}
