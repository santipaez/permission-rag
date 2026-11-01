import unittest
from pathlib import Path

from permrag import (
    Chunk,
    Document,
    ExtractiveLLM,
    Identity,
    KeywordStore,
    Rag,
    build_chunks,
    chunk_document,
    is_visible,
    load_documents,
)

CORPUS = Path(__file__).resolve().parents[1] / "data" / "corpus"


def make_chunk(doc_id: str, text: str, groups: set[str], level: int = 0) -> Chunk:
    return Chunk(
        id=f"{doc_id}#0",
        doc_id=doc_id,
        title=doc_id,
        groups=frozenset(groups),
        level=level,
        text=text,
        ordinal=0,
    )


class VisibilityTests(unittest.TestCase):
    def test_requires_shared_group(self):
        chunk = make_chunk("finance", "revenue", {"finance"}, level=1)
        self.assertFalse(is_visible(chunk, Identity("santi", frozenset({"eng"}), 3)))
        self.assertTrue(is_visible(chunk, Identity("ana", frozenset({"finance"}), 1)))

    def test_requires_clearance_level(self):
        chunk = make_chunk("hr", "bands", {"hr"}, level=4)
        self.assertFalse(is_visible(chunk, Identity("ana", frozenset({"hr"}), 2)))
        self.assertTrue(is_visible(chunk, Identity("ana", frozenset({"hr"}), 4)))

    def test_no_groups_means_nothing_visible(self):
        chunk = make_chunk("eng", "runbook", {"eng"}, level=0)
        self.assertFalse(is_visible(chunk, Identity("nobody", frozenset(), 5)))


class PreFilterTests(unittest.TestCase):
    def test_restricted_match_never_leaks(self):
        chunks = [
            make_chunk("secret", "redis secret access key rotation procedure", {"hr"}, level=4),
            make_chunk("public", "onboarding checklist for new starters", {"eng"}, level=1),
        ]
        store = KeywordStore(chunks)

        hits = store.search("redis secret access key rotation", Identity("santi", frozenset({"eng"}), 2), k=5)
        self.assertNotIn("secret", {hit.chunk.doc_id for hit in hits})

        allowed = store.search("redis secret access key rotation", Identity("hr", frozenset({"hr"}), 4), k=5)
        self.assertEqual(allowed[0].chunk.doc_id, "secret")

    def test_filtering_before_ranking_preserves_recall(self):
        """Ten restricted docs outrank the visible one, but the caller still
        gets the visible answer. Post-filtering would have returned nothing."""
        chunks = [make_chunk(f"secret{i}", "redis key rotation", {"hr"}, level=4) for i in range(10)]
        chunks.append(make_chunk("visible", "redis key rotation for engineering", {"eng"}, level=1))
        store = KeywordStore(chunks)

        hits = store.search("redis key rotation", Identity("santi", frozenset({"eng"}), 2), k=3)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].chunk.doc_id, "visible")


class ChunkingTests(unittest.TestCase):
    def test_long_document_splits(self):
        doc = Document(id="d", title="d", groups=frozenset({"eng"}), level=0, text=("paragraph " * 300).strip())
        chunks = chunk_document(doc, max_chars=500, overlap=50)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk.text) <= 500 for chunk in chunks))

    def test_short_document_stays_whole(self):
        doc = Document(id="d", title="d", groups=frozenset({"eng"}), level=0, text="one short paragraph")
        self.assertEqual(len(chunk_document(doc)), 1)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        chunks = [
            make_chunk("runbook", "Rotate the access key during a maintenance window and restart one instance first.", {"eng"}, level=1),
            make_chunk("finance", "The revenue forecast closes on the tenth business day of the quarter.", {"finance"}, level=3),
        ]
        self.rag = Rag(KeywordStore(chunks), ExtractiveLLM(), k=3, min_score=0.3)

    def test_answers_with_citation(self):
        answer = self.rag.ask("when do I rotate the access key", Identity("s", frozenset({"eng"}), 2))
        self.assertFalse(answer.refused)
        self.assertTrue(answer.citations)
        self.assertIn("[1]", answer.text)

    def test_refuses_when_retrieval_is_empty(self):
        answer = self.rag.ask("what is the capital of France", Identity("s", frozenset({"eng"}), 2))
        self.assertTrue(answer.refused)

    def test_refuses_restricted_topic(self):
        answer = self.rag.ask("when does the revenue forecast close", Identity("s", frozenset({"eng"}), 2))
        self.assertTrue(answer.refused)
        self.assertNotIn("finance", {citation.doc_id for citation in answer.citations})


class CorpusTests(unittest.TestCase):
    def test_sample_corpus_loads(self):
        docs = load_documents(CORPUS)
        self.assertGreaterEqual(len(docs), 5)
        self.assertTrue(all(doc.groups for doc in docs), "every document must declare groups")
        self.assertGreater(len(build_chunks(docs)), len(docs))


if __name__ == "__main__":
    unittest.main()
