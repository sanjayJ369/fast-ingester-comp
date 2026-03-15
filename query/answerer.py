"""
query/answerer.py — LLM-agnostic parallel answer generator.

Supports multiple backends:
  - ollama  (local, free, default)
  - anthropic (Claude API)

Fires all 15 answer requests concurrently using asyncio.
Each call receives the top FINAL_TOP_K retrieved chunks as context.
"""

import asyncio
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import aiohttp

from retrieval.retriever import RetrievedChunk
from config import (
    ANTHROPIC_API_KEY, LLM_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE,
    LLM_BACKEND, OLLAMA_BASE_URL, OLLAMA_MODEL,
)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Answer:
    question:     str
    answer:       str
    doc_name:     str
    page_numbers: list[int]
    confidence:   str = "high"
    chunks_used:  list[str] = field(default_factory=list)


# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise document QA system. Answer questions
using ONLY the provided context chunks. Be concise and factual.

Always respond in this exact JSON format:
{
  "answer": "<your answer here>",
  "doc_name": "<filename of the most relevant source document>",
  "page_numbers": [<list of page numbers cited, e.g. 1, 3>],
  "confidence": "<high|medium|low>"
}

Rules:
- If the answer is not in the context, set answer to "NOT_FOUND" and confidence to "low"
- Keep answers concise — 1-3 sentences max
- Only cite page numbers that directly support the answer
- doc_name must exactly match one of the provided filenames"""


def _build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Chunk {i} | File: {chunk.filename} | Page: {chunk.page_num}]\n"
            f"{chunk.text}"
        )
    context = "\n\n---\n\n".join(context_parts)
    return f"Context:\n{context}\n\nQuestion: {question}"


def _parse_llm_response(raw: str, question: str, chunks: list[RetrievedChunk]) -> Answer:
    """Parse JSON from LLM response, with fallback."""
    try:
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        parsed = json.loads(text)
        ans = parsed.get("answer", "NOT_FOUND")
        if isinstance(ans, list):
            ans = " ".join(map(str, ans))
        else:
            ans = str(ans)

        doc = parsed.get("doc_name", "")
        if isinstance(doc, list):
            doc = doc[0] if doc else ""
        doc = str(doc)

        return Answer(
            question=question,
            answer=ans,
            doc_name=doc,
            page_numbers=parsed.get("page_numbers", []),
            confidence=parsed.get("confidence", "low"),
            chunks_used=[c.chunk_id for c in chunks],
        )
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"[ANSWERER] JSON parse failed for '{question[:40]}': {e}")
        return Answer(
            question=question,
            answer="PARSE_ERROR",
            doc_name=chunks[0].filename if chunks else "",
            page_numbers=[chunks[0].page_num] if chunks else [],
            confidence="low",
        )


# ── Backend: Ollama ──────────────────────────────────────────────────────────

class OllamaBackend:
    def __init__(self):
        self.base_url = OLLAMA_BASE_URL
        self.model = OLLAMA_MODEL
        self._session = None

    async def _get_session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def answer_question(
        self, question: str, chunks: list[RetrievedChunk]
    ) -> Answer:
        if not chunks:
            return Answer(question=question, answer="NOT_FOUND", doc_name="",
                          page_numbers=[], confidence="low")

        session = await self._get_session()
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(question, chunks)},
            ],
            "stream": False,
            "options": {
                "temperature": LLM_TEMPERATURE,
                "num_predict": LLM_MAX_TOKENS,
            },
        }

        try:
            async with session.post(
                f"{self.base_url}/api/chat", json=payload
            ) as resp:
                data = await resp.json()
                raw = data["message"]["content"]
                return _parse_llm_response(raw, question, chunks)
        except Exception as e:
            print(f"[ANSWERER] Ollama error: {e}")
            return Answer(question=question, answer="LLM_ERROR", doc_name="",
                          page_numbers=[], confidence="low")

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()


# ── Backend: Anthropic (Claude) ──────────────────────────────────────────────

class AnthropicBackend:
    def __init__(self):
        import anthropic
        self.client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

    async def answer_question(
        self, question: str, chunks: list[RetrievedChunk]
    ) -> Answer:
        if not chunks:
            return Answer(question=question, answer="NOT_FOUND", doc_name="",
                          page_numbers=[], confidence="low")

        try:
            response = await self.client.messages.create(
                model=LLM_MODEL,
                max_tokens=LLM_MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": _build_user_prompt(question, chunks)}],
            )
            raw = response.content[0].text
            return _parse_llm_response(raw, question, chunks)

        except Exception as e:
            print(f"[ANSWERER] Anthropic error: {e}")
            return Answer(question=question, answer="API_ERROR", doc_name="",
                          page_numbers=[], confidence="low")

    async def close(self):
        pass


# ── Factory ──────────────────────────────────────────────────────────────────

def get_backend():
    if LLM_BACKEND == "anthropic":
        return AnthropicBackend()
    else:
        return OllamaBackend()


# ── Batch answerer (all 15 questions concurrently) ────────────────────────────

async def answer_all(
    questions: list[str],
    retrieved: list[list[RetrievedChunk]],
) -> list[Answer]:
    """
    Fire all LLM calls concurrently.

    Args:
        questions:  list of 15 question strings
        retrieved:  list of 15 retrieved chunk lists (aligned with questions)

    Returns:
        list of 15 Answer objects
    """
    backend = get_backend()
    print(f"[ANSWERER] Using backend: {LLM_BACKEND}")

    tasks = [
        backend.answer_question(q, chunks)
        for q, chunks in zip(questions, retrieved)
    ]
    answers = await asyncio.gather(*tasks)
    await backend.close()

    print(f"[ANSWERER] {len(answers)} answers generated")
    for i, a in enumerate(answers):
        status = "OK" if a.confidence != "low" else "??"
        print(f"  Q{i+1:02d} [{status} {a.confidence}] {a.question[:50]}...")

    return list(answers)
