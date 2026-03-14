package parser

import (
	"archive/zip"
	"encoding/xml"
	"io"
	"path/filepath"
	"strings"
)

// Minimal DOCX XML structures for word/document.xml
type docxBody struct {
	Paragraphs []docxParagraph `xml:"body>p"`
	Tables     []docxTable     `xml:"body>tbl"`
}

type docxParagraph struct {
	Properties docxParagraphProps `xml:"pPr"`
	Runs       []docxRun          `xml:"r"`
}

type docxParagraphProps struct {
	Style docxStyle `xml:"pStyle"`
}

type docxStyle struct {
	Val string `xml:"val,attr"`
}

type docxRun struct {
	Text []docxText `xml:"t"`
}

type docxText struct {
	Content string `xml:",chardata"`
}

type docxTable struct {
	Rows []docxRow `xml:"tr"`
}

type docxRow struct {
	Cells []docxCell `xml:"tc"`
}

type docxCell struct {
	Paragraphs []docxParagraph `xml:"p"`
}

func parseDOCX(path string) (ParsedDocument, error) {
	r, err := zip.OpenReader(path)
	if err != nil {
		return ParsedDocument{}, err
	}
	defer r.Close()

	base := filepath.Base(path)
	stem := strings.TrimSuffix(base, filepath.Ext(base))

	// Find word/document.xml
	var docFile *zip.File
	for _, f := range r.File {
		if f.Name == "word/document.xml" {
			docFile = f
			break
		}
	}
	if docFile == nil {
		return ParsedDocument{
			DocID:    stem,
			Filename: base,
			Pages:    []string{""},
		}, nil
	}

	rc, err := docFile.Open()
	if err != nil {
		return ParsedDocument{}, err
	}
	defer rc.Close()

	data, err := io.ReadAll(rc)
	if err != nil {
		return ParsedDocument{}, err
	}

	var body docxBody
	if err := xml.Unmarshal(data, &body); err != nil {
		return ParsedDocument{}, err
	}

	// Extract paragraphs, splitting on headings (matching Python behavior)
	var sections []string
	var current []string

	for _, para := range body.Paragraphs {
		text := extractParagraphText(para)
		if text == "" {
			continue
		}

		isHeading := strings.HasPrefix(strings.ToLower(para.Properties.Style.Val), "heading")
		if isHeading {
			if len(current) > 0 {
				sections = append(sections, strings.Join(current, "\n"))
				current = nil
			}
			current = append(current, "## "+text)
		} else {
			current = append(current, text)
		}
	}
	if len(current) > 0 {
		sections = append(sections, strings.Join(current, "\n"))
	}

	// Extract tables
	for _, table := range body.Tables {
		var rows []string
		for _, row := range table.Rows {
			var cells []string
			for _, cell := range row.Cells {
				var cellText []string
				for _, p := range cell.Paragraphs {
					t := extractParagraphText(p)
					if t != "" {
						cellText = append(cellText, t)
					}
				}
				cells = append(cells, strings.Join(cellText, " "))
			}
			rows = append(rows, strings.Join(cells, "\t"))
		}
		if len(rows) > 0 {
			sections = append(sections, strings.Join(rows, "\n"))
		}
	}

	return ParsedDocument{
		DocID:    stem,
		Filename: base,
		Pages:    sections,
	}, nil
}

func extractParagraphText(para docxParagraph) string {
	var parts []string
	for _, run := range para.Runs {
		for _, t := range run.Text {
			if t.Content != "" {
				parts = append(parts, t.Content)
			}
		}
	}
	return strings.TrimSpace(strings.Join(parts, ""))
}
