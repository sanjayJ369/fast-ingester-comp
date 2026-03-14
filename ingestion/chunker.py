"""
ingestion/chunker.py — Recursive character-level chunker.

Splits ParsedDocuments into overlapping chunks that preserve
paragraph and sentence boundaries as much as possible.
"""

from dataclasses import dataclass, field
from typing import Generator
from ingestion.parser import ParsedDocument
from config import CHUNK_SIZE, CHUNK_OVERLAP


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    chunk_id:  str          # "{doc_id}__chunk_{n}"
    doc_id:    str
    filename:  str
    text:      str
    page_num:  int          # which page this chunk came from
    chunk_idx: int          # index within the document
    metadata:  dict = field(default_factory=dict)


# ── Splitter ──────────────────────────────────────────────────────────────────

def _split_text(text: str, size: int, overlap: int) -> list[str]:
    """
    Recursive character splitter. Tries to split on:
      1. Double newline (paragraph)
      2. Single newline
      3. Period/sentence boundary
      4. Hard character limit (last resort)
    """
    if len(text) <= size:
        return [text]

    # find the best split point near `size`
    split_chars = ["\n\n", "\n", ". ", " "]
    for sep in split_chars:
        idx = text.rfind(sep, 0, size)
        if idx > size // 2:          # only use if we're at least halfway
            split_at = idx + len(sep)
            head = text[:split_at].strip()
            tail = text[max(0, split_at - overlap):].strip()
            return [head] + _split_text(tail, size, overlap)

    # hard cut
    head = text[:size]
    tail = text[max(0, size - overlap):]
    return [head] + _split_text(tail, size, overlap)


# ── Public API ────────────────────────────────────────────────────────────────

def chunk_document(doc: ParsedDocument) -> list[Chunk]:
    """
    Chunk a single ParsedDocument into overlapping Chunk objects.
    """
    chunks: list[Chunk] = []
    global_idx = 0

    for page_num, page_text in enumerate(doc.pages):
        if not page_text.strip():
            continue

        splits = _split_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP)

        for split in splits:
            if not split.strip():
                continue
            chunks.append(Chunk(
                chunk_id  = f"{doc.doc_id}__chunk_{global_idx}",
                doc_id    = doc.doc_id,
                filename  = doc.filename,
                text      = split.strip(),
                page_num  = page_num + 1,   # 1-indexed
                chunk_idx = global_idx,
                metadata  = {**doc.metadata, "page": page_num + 1},
            ))
            global_idx += 1

    return chunks


def chunk_all(docs: list[ParsedDocument]) -> list[Chunk]:
    """
    Chunk all documents. Pure CPU work — no async needed here,
    called from ingestion pipeline after parse_all().
    """
    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))

    print(f"[CHUNKER] ✅ {len(all_chunks)} chunks from {len(docs)} documents "
          f"(avg {len(all_chunks)//max(len(docs),1)} chunks/doc)")
    return all_chunks
