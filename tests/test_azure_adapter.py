import unittest
from types import SimpleNamespace

from permrag import AzureOpenAILLM, Chunk, Identity, KeywordStore, Rag
from permrag.llm import SYSTEM_PROMPT


class FakeCompletions:
    def __init__(self, content: str | None) -> None:
        self.calls: list[dict] = []
        self._content = content

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self._content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def fake_client(content: str | None) -> tuple[SimpleNamespace, FakeCompletions]:
    completions = FakeCompletions(content)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


def chunk(doc_id: str, text: str, groups: set[str], level: int = 0) -> Chunk:
    return Chunk(
        id=f"{doc_id}#0",
        doc_id=doc_id,
        title=doc_id,
        groups=frozenset(groups),
        level=level,
        text=text,
        ordinal=0,
    )


class AzureOpenAILLMTests(unittest.TestCase):
    def test_sends_numbered_sources_and_grounding_prompt(self):
        client, completions = fake_client("Rotate the key during a maintenance window. [1]")
        llm = AzureOpenAILLM(client=client, deployment="gpt-4o-mini")

        answer = llm.answer(
            "how do I rotate the key",
            [chunk("redis-runbook", "Rotate the key during a maintenance window.", {"eng"})],
        )

        self.assertEqual(answer, "Rotate the key during a maintenance window. [1]")
        request = completions.calls[0]
        self.assertEqual(request["model"], "gpt-4o-mini")
        self.assertEqual(request["temperature"], 0)
        self.assertEqual(request["messages"][0], {"role": "system", "content": SYSTEM_PROMPT})
        self.assertIn("[1] redis-runbook", request["messages"][1]["content"])

    def test_empty_model_output_is_returned_as_empty(self):
        client, _ = fake_client(None)
        llm = AzureOpenAILLM(client=client, deployment="gpt-4o-mini")
        self.assertEqual(llm.answer("q", [chunk("d", "text", {"eng"})]), "")

    def test_rag_refuses_when_model_returns_nothing(self):
        client, _ = fake_client("")
        store = KeywordStore([chunk("runbook", "Rotate the access key during maintenance.", {"eng"})])
        rag = Rag(store, AzureOpenAILLM(client=client, deployment="gpt-4o-mini"), min_score=0.1)

        answer = rag.ask("how do I rotate the access key", Identity("s", frozenset({"eng"}), 2))
        self.assertTrue(answer.refused)


if __name__ == "__main__":
    unittest.main()
