"""Run the golden set against the offline pipeline.

    retrieval_hit  the expected document is among the cited sources
    refusal        the assistant declined when it should have
    leak           a document the caller must not see reached the answer

A leak fails the run with exit status 1, so the script can gate CI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from permrag import ExtractiveLLM, Identity, KeywordStore, Rag, build_chunks, load_documents  # noqa: E402

IDENTITIES = {
    "santi": Identity(user="santi", groups=frozenset({"eng", "people"}), level=2),
    "ana": Identity(user="ana", groups=frozenset({"finance"}), level=3),
    "lia": Identity(user="lia", groups=frozenset({"security", "eng"}), level=2),
}


def load_cases(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    store = KeywordStore(build_chunks(load_documents(ROOT / "data" / "corpus")))
    rag = Rag(store, ExtractiveLLM(), k=4, min_score=0.6)
    cases = load_cases(ROOT / "evals" / "golden.jsonl")

    answerable = [c for c in cases if not c.get("expect_refusal")]
    refusable = [c for c in cases if c.get("expect_refusal")]
    hits = refusals = leaks = 0

    print(f"{'id':5} {'user':6} {'refused':8} {'hit':5} {'leak':5} question")
    print("-" * 88)

    for case in cases:
        answer = rag.ask(case["question"], IDENTITIES[case["user"]])
        cited = {citation.doc_id for citation in answer.citations}
        forbidden = set(case.get("must_not_cite", []))
        leaked = bool((cited | {h.chunk.doc_id for h in answer.hits}) & forbidden)

        if case in answerable:
            hit = case["expect_doc"] in cited
            hits += hit
            verdict = str(hit)
        else:
            refusals += answer.refused
            verdict = "-"

        leaks += leaked
        print(f"{case['id']:5} {case['user']:6} {str(answer.refused):8} {verdict:5} {str(leaked):5} {case['question']}")

    print("-" * 88)
    print(f"retrieval hit rate    {hits}/{len(answerable)}")
    print(f"refusal accuracy      {refusals}/{len(refusable)}")
    print(f"acl leaks             {leaks}")
    return 1 if leaks else 0


if __name__ == "__main__":
    raise SystemExit(main())
