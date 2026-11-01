"""Measure what happens to recall when access control is applied after ranking.

Both strategies use the same documents and the same BM25 implementation. The
only difference is the order: filter then rank, or rank then filter.

The corpus is synthetic on purpose. Each topic contains a fixed number of
visible documents and a growing number of restricted ones that share the same
vocabulary, which is the realistic case: a restricted runbook that answers the
question better than the visible one.

    python benchmarks/recall_sweep.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from permrag import Chunk, Document, Identity, KeywordStore, build_chunks  # noqa: E402

BASE_TOPICS = [
    "redis", "onboarding", "security", "backup", "roadmap",
    "monitoring", "identity", "deployment", "networking", "billing",
]
# Alphabetic only. Digits in a topic name become tokens with their own term
# frequency, which then decides the ranking and contaminates the measurement.
VARIANTS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
TOPICS = [f"{base} {variant}" for base in BASE_TOPICS for variant in VARIANTS]

VISIBLE_PER_TOPIC = 3
K = 3
RESTRICTED_STEPS = [0, 1, 2, 3, 5, 8, 12, 20, 35]

ENGINEER = Identity(user="engineer", groups=frozenset({"eng"}), level=1)
EVERYTHING = Identity(user="everything", groups=frozenset({"eng", "finance"}), level=9)


def build_documents(restricted_per_topic: int) -> list[Document]:
    """All documents in a topic share the same vocabulary, so the ranker has no
    way to prefer the visible one. That is the worst case for post-filtering
    and the point of the experiment."""
    docs: list[Document] = []
    for topic in TOPICS:
        body = (
            f"{topic} runbook. This {topic} runbook covers the {topic} steps. "
            f"Follow the {topic} procedure carefully and review the {topic} procedure afterwards."
        )
        for index in range(VISIBLE_PER_TOPIC):
            docs.append(
                Document(
                    id=f"{topic}/visible/{index}",
                    title=f"{topic} runbook",
                    groups=frozenset({"eng"}),
                    level=1,
                    # Identical length to the restricted copy: any difference in
                    # token count would move the BM25 length normalisation and
                    # decide the ranking for the wrong reason.
                    text=f"{body} Note A.",
                )
            )
        for index in range(restricted_per_topic):
            docs.append(
                Document(
                    id=f"{topic}/restricted/{index}",
                    title=f"{topic} runbook",
                    groups=frozenset({"finance"}),
                    level=1,
                    text=f"{body} Note A.",
                )
            )
    # Equal scores tie, and a stable sort keeps insertion order, which would put
    # the visible documents first and flatter post-filtering. Shuffle with a
    # fixed seed so ties break neutrally and the run stays reproducible.
    random.Random(0).shuffle(docs)
    return docs


def evaluate(restricted_per_topic: int) -> dict[str, float]:
    chunks = build_chunks(build_documents(restricted_per_topic))

    pre_filter = KeywordStore(chunks)
    rank_everything = KeywordStore(chunks)

    queries = [(topic, f"{topic} runbook procedure") for topic in TOPICS]

    pre_hits = post_hits = empty = 0

    for topic, query in queries:
        visible_targets = {
            f"{topic}/visible/{index}" for index in range(VISIBLE_PER_TOPIC)
        }

        pre = pre_filter.search(query, ENGINEER, k=K)
        pre_hits += bool(visible_targets & {hit.chunk.doc_id for hit in pre})

        # Rank over the whole corpus with an identity that sees everything, then
        # drop what the engineer may not read. This is the pattern under test.
        ranked = rank_everything.search(query, EVERYTHING, k=K)
        visible = [hit for hit in ranked if hit.chunk.doc_id in visible_targets]
        post_hits += bool(visible)
        empty += not visible

    total = len(queries)
    return {
        "recall_pre": pre_hits / total,
        "recall_post": post_hits / total,
        "empty_rate": empty / total,
    }


def render_svg(points: list[tuple[int, float, float]]) -> str:
    width, height = 720, 420
    left, right, top, bottom = 70, 40, 50, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_x = max(point[0] for point in points) or 1

    def x(value: float) -> float:
        return left + (value / max_x) * plot_w

    def y(value: float) -> float:
        return top + (1 - value) * plot_h

    pre_line = " ".join(f"{x(p[0]):.1f},{y(p[1]):.1f}" for p in points)
    post_line = " ".join(f"{x(p[0]):.1f},{y(p[2]):.1f}" for p in points)
    markers_pre = "".join(
        f'<circle cx="{x(p[0]):.1f}" cy="{y(p[1]):.1f}" r="3.5" fill="#16a34a"/>' for p in points
    )
    markers_post = "".join(
        f'<circle cx="{x(p[0]):.1f}" cy="{y(p[2]):.1f}" r="3.5" fill="#dc2626"/>' for p in points
    )

    grid = ""
    for step in (0, 0.25, 0.5, 0.75, 1.0):
        grid += (
            f'<line x1="{left}" y1="{y(step):.1f}" x2="{width - right}" y2="{y(step):.1f}" '
            f'stroke="#e2e8f0"/><text x="{left - 12}" y="{y(step) + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="#64748b">{int(step * 100)}%</text>'
        )

    ticks = "".join(
        f'<text x="{x(p[0]):.1f}" y="{height - bottom + 20}" text-anchor="middle" font-size="11" '
        f'fill="#64748b">{p[0]}</text>'
        for p in points
    )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="t">
  <title id="t">Recall by restricted documents per topic</title>
  <rect width="{width}" height="{height}" fill="#ffffff"/>
  <text x="{left}" y="26" font-family="Segoe UI, Arial, sans-serif" font-size="16" font-weight="700" fill="#0f172a">Recall@3 as restricted documents grow</text>
  <text x="{left}" y="44" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#475569">Synthetic corpus, {len(TOPICS)} topics, {VISIBLE_PER_TOPIC} visible documents each, identical BM25 ranking.</text>
  {grid}
  <text x="20" y="{top + plot_h / 2:.0f}" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#475569" transform="rotate(-90 20 {top + plot_h / 2:.0f})" text-anchor="middle">recall</text>
  <text x="{left + plot_w / 2:.0f}" y="{height - 20}" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#475569" text-anchor="middle">restricted documents per topic</text>
  <polyline points="{pre_line}" fill="none" stroke="#16a34a" stroke-width="2.5"/>
  <polyline points="{post_line}" fill="none" stroke="#dc2626" stroke-width="2.5"/>
  {markers_pre}{markers_post}
  {ticks}
  <rect x="{width - right - 190}" y="{top + 8}" width="180" height="52" rx="8" fill="#f8fafc" stroke="#e2e8f0"/>
  <circle cx="{width - right - 172}" cy="{top + 26}" r="4" fill="#16a34a"/>
  <text x="{width - right - 160}" y="{top + 30}" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#0f172a">filter, then rank</text>
  <circle cx="{width - right - 172}" cy="{top + 46}" r="4" fill="#dc2626"/>
  <text x="{width - right - 160}" y="{top + 50}" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#0f172a">rank, then filter</text>
</svg>
"""


def main() -> None:
    print(f"{'restricted':>10} {'recall pre':>11} {'recall post':>12} {'empty post':>11}")
    print("-" * 48)
    points: list[tuple[int, float, float]] = []

    for step in RESTRICTED_STEPS:
        metrics = evaluate(step)
        points.append((step, metrics["recall_pre"], metrics["recall_post"]))
        print(
            f"{step:>10} {metrics['recall_pre']:>11.0%} {metrics['recall_post']:>12.0%} "
            f"{metrics['empty_rate']:>11.0%}"
        )

    target = ROOT / "docs" / "recall-sweep.svg"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_svg(points), encoding="utf-8")
    print(f"\nwrote {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
