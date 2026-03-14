package parser

import (
	"os"
	"path/filepath"
	"strings"

	"github.com/PuerkitoBio/goquery"
)

func parseHTML(path string) (ParsedDocument, error) {
	f, err := os.Open(path)
	if err != nil {
		return ParsedDocument{}, err
	}
	defer f.Close()

	base := filepath.Base(path)
	stem := strings.TrimSuffix(base, filepath.Ext(base))

	doc, err := goquery.NewDocumentFromReader(f)
	if err != nil {
		return ParsedDocument{}, err
	}

	// Remove scripts, styles, nav, footer (matching Python behavior)
	doc.Find("script, style, nav, footer").Remove()

	text := doc.Text()

	// Clean up: split lines, trim, rejoin non-empty
	lines := strings.Split(text, "\n")
	var cleaned []string
	for _, l := range lines {
		l = strings.TrimSpace(l)
		if l != "" {
			cleaned = append(cleaned, l)
		}
	}

	content := strings.Join(cleaned, "\n")

	return ParsedDocument{
		DocID:    stem,
		Filename: base,
		Pages:    []string{content},
	}, nil
}
