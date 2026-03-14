package parser

import (
	"fmt"
	"path/filepath"
	"strings"

	"github.com/xuri/excelize/v2"
)

func parseExcel(path string) (ParsedDocument, error) {
	f, err := excelize.OpenFile(path)
	if err != nil {
		return ParsedDocument{}, err
	}
	defer f.Close()

	base := filepath.Base(path)
	stem := strings.TrimSuffix(base, filepath.Ext(base))

	var pages []string

	for _, sheetName := range f.GetSheetList() {
		rows, err := f.GetRows(sheetName)
		if err != nil {
			continue
		}

		var rowTexts []string
		for _, row := range rows {
			rowText := strings.Join(row, "\t")
			if strings.TrimSpace(rowText) != "" {
				rowTexts = append(rowTexts, rowText)
			}
		}

		// Chunk into 50-row pages (matching Python behavior)
		for i := 0; i < len(rowTexts); i += 50 {
			end := i + 50
			if end > len(rowTexts) {
				end = len(rowTexts)
			}
			header := fmt.Sprintf("[Sheet: %s]\n", sheetName)
			pages = append(pages, header+strings.Join(rowTexts[i:end], "\n"))
		}
	}

	return ParsedDocument{
		DocID:    stem,
		Filename: base,
		Pages:    pages,
	}, nil
}
