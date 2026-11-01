"""Answer generation.

`ExtractiveLLM` is the offline default: it builds an answer only from
retrieved sentences, which keeps the eval harness honest and the repo
runnable without a cloud account. `AzureOpenAILLM` is the production path.
"""

from __future__ import annotations

import re
from typing import Protocol, Sequence

from .core import Chunk

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


class LLM(Protocol):
    def answer(self, question: str, contexts: Sequence[Chunk]) -> str: ...


class ExtractiveLLM:
    """Picks the sentences that best overlap the question. No hallucination
    is possible because every word comes from a retrieved chunk."""

    def __init__(self, max_sentences: int = 3) -> None:
        self._max_sentences = max_sentences

    def answer(self, question: str, contexts: Sequence[Chunk]) -> str:
        terms = {token for token in re.findall(r"[a-z0-9]+", question.lower()) if len(token) > 2}
        candidates: list[tuple[float, str, int]] = []

        for index, chunk in enumerate(contexts, start=1):
            body = " ".join(line for line in chunk.text.splitlines() if not line.startswith("#"))
            for sentence in _SENTENCE.split(body):
                sentence = sentence.strip()
                if len(sentence) < 20:
                    continue
                overlap = len(terms & set(re.findall(r"[a-z0-9]+", sentence.lower())))
                if overlap:
                    candidates.append((overlap, sentence, index))

        if not candidates:
            return ""

        candidates.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
        chosen = candidates[: self._max_sentences]
        chosen.sort(key=lambda item: item[2])
        parts = [f"{sentence} [{index}]" for _, sentence, index in chosen]
        return " ".join(parts)


class AzureOpenAILLM:
    """Chat completions against an Azure OpenAI deployment.

    Needs the `azure` extra. Authenticates with Entra ID through
    DefaultAzureCredential, so no API key is read from the environment.
    """

    _SCOPE = "https://cognitiveservices.azure.com/.default"

    def __init__(self, endpoint: str, deployment: str, api_version: str = "2024-10-21") -> None:
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider
        from openai import AzureOpenAI

        token_provider = get_bearer_token_provider(DefaultAzureCredential(), self._SCOPE)
        self._client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_version=api_version,
            azure_ad_token_provider=token_provider,
        )
        self._deployment = deployment

    def answer(self, question: str, contexts: Sequence[Chunk]) -> str:
        blocks = "\n\n".join(
            f"[{index}] {chunk.title}\n{chunk.text}" for index, chunk in enumerate(contexts, start=1)
        )
        response = self._client.chat.completions.create(
            model=self._deployment,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer only from the numbered sources. Cite them as [n]. "
                        "If the sources do not contain the answer, reply exactly: "
                        "I don't have access to that information."
                    ),
                },
                {"role": "user", "content": f"Sources:\n{blocks}\n\nQuestion: {question}"},
            ],
        )
        return (response.choices[0].message.content or "").strip()
