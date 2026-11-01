"""Keyword retrieval (BM25) with access control applied before ranking.

The ordering matters. Filtering after ranking leaks in two ways:

1. a caller can probe which documents exist by watching the result count,
2. recall collapses when the top-k is dominated by documents the caller
   cannot see, because the visible ones were ranked below the cut.

So `search` filters candidates first and scores only what the caller may read.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from .core import Chunk, Hit, Identity, is_visible

_TOKEN = re.compile(r"[a-z0-9]+")

# Function words carry no retrieval signal and, left in, they inflate every
# score enough that the refusal threshold stops working.
STOPWORDS = frozenset(
    """a an and are as at be been by can could did do does for from had has have
    how i if in into is it its just more most much my no not of on or our out
    over should so some such than that the their them then there these they this
    those to too up was we were what when where which who why will with would you
    your""".split()
)


def tokenize(text: str) -> list[str]:
    return [token for token in _TOKEN.findall(text.lower()) if token not in STOPWORDS]


class KeywordStore:
    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self._tokens = [tokenize(f"{chunk.title} {chunk.text}") for chunk in chunks]
        self._lengths = [len(tokens) for tokens in self._tokens]
        self._avg_length = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0

        df: Counter[str] = Counter()
        for tokens in self._tokens:
            df.update(set(tokens))
        total = len(chunks)
        self._idf = {term: math.log(1 + (total - count + 0.5) / (count + 0.5)) for term, count in df.items()}

    def __len__(self) -> int:
        return len(self._chunks)

    def search(self, query: str, identity: Identity, k: int = 5) -> list[Hit]:
        terms = tokenize(query)
        if not terms:
            return []

        visible = [
            (index, chunk)
            for index, chunk in enumerate(self._chunks)
            if is_visible(chunk, identity)
        ]
        if not visible:
            return []

        k1, b = 1.5, 0.75
        scored: list[Hit] = []
        for index, chunk in visible:
            tokens = self._tokens[index]
            if not tokens:
                continue
            frequencies = Counter(tokens)
            length = self._lengths[index] or 1
            score = 0.0
            for term in terms:
                tf = frequencies.get(term)
                if not tf:
                    continue
                idf = self._idf.get(term, 0.0)
                denominator = tf + k1 * (1 - b + b * length / (self._avg_length or 1))
                score += idf * (tf * (k1 + 1)) / denominator
            if score > 0:
                scored.append(Hit(chunk=chunk, score=score))

        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:k]
