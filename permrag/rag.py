"""Question answering with access control applied before retrieval ranks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .core import Chunk, Hit, Identity
from .llm import LLM
from .store import tokenize

REFUSAL = "I don't have access to that information."


class Retriever(Protocol):
    def search(self, query: str, identity: Identity, k: int = 5) -> list[Hit]: ...


@dataclass(frozen=True)
class Citation:
    chunk_id: str
    doc_id: str
    title: str


@dataclass(frozen=True)
class Answer:
    text: str
    refused: bool
    citations: list[Citation] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)


class Rag:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLM,
        k: int = 4,
        min_score: float = 0.6,
        min_terms: int = 2,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._k = k
        self._min_score = min_score
        self._min_terms = min_terms

    def ask(self, question: str, identity: Identity) -> Answer:
        hits = self._retriever.search(question, identity, k=self._k)
        if not hits or not self._has_evidence(question, hits):
            return Answer(text=REFUSAL, refused=True, hits=hits)

        contexts: list[Chunk] = [hit.chunk for hit in hits]
        text = self._llm.answer(question, contexts).strip()
        if not text:
            return Answer(text=REFUSAL, refused=True, hits=hits)

        citations = [Citation(chunk_id=c.id, doc_id=c.doc_id, title=c.title) for c in contexts]
        return Answer(text=text, refused=False, citations=citations, hits=hits)

    def _has_evidence(self, question: str, hits: list[Hit]) -> bool:
        """A weak score or a single shared word is not evidence. The top chunk
        must score above the threshold and cover enough distinct query terms."""
        if hits[0].score < self._min_score:
            return False
        query_terms = set(tokenize(question))
        top = hits[0].chunk
        matched = query_terms & set(tokenize(f"{top.title} {top.text}"))
        return len(matched) >= min(self._min_terms, len(query_terms))
