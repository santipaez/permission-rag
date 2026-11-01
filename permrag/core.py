"""Core types: documents, chunks, identities and the visibility rule."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    groups: frozenset[str]
    level: int
    text: str


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    title: str
    groups: frozenset[str]
    level: int
    text: str
    ordinal: int


@dataclass(frozen=True)
class Identity:
    """The caller. In production this comes from an Entra ID token;
    groups map to App Roles or directory groups."""

    user: str
    groups: frozenset[str] = field(default_factory=frozenset)
    level: int = 0


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


def is_visible(chunk: Chunk, identity: Identity) -> bool:
    """A chunk is visible when the caller shares at least one group and
    meets the document's clearance level."""
    return bool(chunk.groups & identity.groups) and identity.level >= chunk.level
