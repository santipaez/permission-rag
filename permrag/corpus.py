"""Load a markdown corpus with front matter and split it into chunks."""

from __future__ import annotations

import re
from pathlib import Path

from .core import Chunk, Document

_FRONT_MATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


def _parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    match = _FRONT_MATTER.match(raw)
    if not match:
        return {}, raw
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip()
    return meta, match.group(2)


def _to_groups(value: str) -> frozenset[str]:
    return frozenset(part.strip() for part in value.split(",") if part.strip())


def load_documents(root: Path) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(root.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        meta, body = _parse_front_matter(raw)
        docs.append(
            Document(
                id=meta.get("id", path.stem),
                title=meta.get("title", path.stem),
                groups=_to_groups(meta.get("groups", "")),
                level=int(meta.get("level", "0")),
                text=body.strip(),
            )
        )
    return docs


def chunk_document(doc: Document, max_chars: int = 700, overlap: int = 120) -> list[Chunk]:
    """Paragraph-aware chunking. Long paragraphs are hard-split with overlap
    so a sentence is never cut in the middle of a heading."""
    paragraphs = [p.strip() for p in doc.text.split("\n\n") if p.strip()]
    pieces: list[str] = []
    buffer = ""

    for paragraph in paragraphs:
        if not buffer:
            buffer = paragraph
        elif len(buffer) + len(paragraph) + 2 <= max_chars:
            buffer = f"{buffer}\n\n{paragraph}"
        else:
            pieces.append(buffer)
            tail = buffer[-overlap:] if overlap and len(buffer) > overlap else ""
            buffer = f"{tail}\n\n{paragraph}" if tail else paragraph

        while len(buffer) > max_chars:
            pieces.append(buffer[:max_chars])
            buffer = buffer[max_chars - overlap :] if overlap else buffer[max_chars:]

    if buffer:
        pieces.append(buffer)

    return [
        Chunk(
            id=f"{doc.id}#{index}",
            doc_id=doc.id,
            title=doc.title,
            groups=doc.groups,
            level=doc.level,
            text=piece.strip(),
            ordinal=index,
        )
        for index, piece in enumerate(pieces)
    ]


def build_chunks(docs: list[Document]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc))
    return chunks
