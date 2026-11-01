"""Small CLI so the repo is demonstrable in one command.

    python -m permrag "how do I rotate the redis credential" --as santi
    python -m permrag "what is the Q3 forecast" --as santi
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .corpus import build_chunks, load_documents
from .core import Identity
from .llm import ExtractiveLLM
from .rag import Rag
from .store import KeywordStore

# Demo identities. In production these come from the validated Entra ID token.
IDENTITIES = {
    "santi": Identity(user="santi", groups=frozenset({"eng", "people"}), level=2),
    "ana": Identity(user="ana", groups=frozenset({"finance"}), level=3),
    "lia": Identity(user="lia", groups=frozenset({"security", "eng"}), level=2),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the permission-aware corpus.")
    parser.add_argument("question")
    parser.add_argument("--as", dest="who", default="santi", choices=sorted(IDENTITIES))
    parser.add_argument("--corpus", default="data/corpus")
    parser.add_argument("-k", type=int, default=4)
    args = parser.parse_args()

    docs = load_documents(Path(args.corpus))
    store = KeywordStore(build_chunks(docs))
    rag = Rag(store, ExtractiveLLM(), k=args.k)

    identity = IDENTITIES[args.who]
    answer = rag.ask(args.question, identity)

    print(f"user      {identity.user}  groups={sorted(identity.groups)}  level={identity.level}")
    print(f"question  {args.question}")
    print(f"answer    {answer.text}")
    if answer.citations:
        for index, citation in enumerate(answer.citations, start=1):
            print(f"  [{index}] {citation.title}  ({citation.doc_id})")
    print(f"refused   {answer.refused}")


if __name__ == "__main__":
    main()
