"""Permission-aware RAG over a private corpus."""

from .core import Chunk, Document, Hit, Identity, is_visible
from .corpus import build_chunks, chunk_document, load_documents
from .llm import AzureOpenAILLM, ExtractiveLLM, LLM
from .rag import REFUSAL, Answer, Citation, Rag
from .store import KeywordStore, tokenize

__all__ = [
    "Answer",
    "AzureOpenAILLM",
    "Chunk",
    "Citation",
    "Document",
    "ExtractiveLLM",
    "Hit",
    "Identity",
    "KeywordStore",
    "LLM",
    "REFUSAL",
    "Rag",
    "build_chunks",
    "chunk_document",
    "is_visible",
    "load_documents",
    "tokenize",
]
