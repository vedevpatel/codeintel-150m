"""Compact, self-contained training data for shipping to a short-lived GPU machine.

Chunk texts are deduplicated into a table (the same function is a positive in one group and a negative in
another) and truncated to what the model can see, so the file is a fraction of the source JSONL.
"""

from __future__ import annotations

import gzip
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from codeintel.training.datasets import TrainingGroup


def pack_groups(
    groups: Sequence[TrainingGroup], path: str | Path, max_doc_chars: int = 2000, max_query_chars: int = 1500,
    max_positives: int = 6, max_negatives: int = 8, per_source_cap: Mapping[str, int] | None = None, seed: int = 13,
) -> dict:
    """Write groups to a gzipped JSON file; returns size statistics.

    `per_source_cap` limits groups kept per source (a low-share source needs far fewer groups than it has).
    """
    import random

    rng = random.Random(seed)
    kept: list[TrainingGroup] = []
    by_source: dict[str, list[TrainingGroup]] = {}
    for g in groups:
        by_source.setdefault(g.source, []).append(g)
    for source, items in by_source.items():
        cap = (per_source_cap or {}).get(source)
        kept += rng.sample(items, cap) if cap is not None and len(items) > cap else items
    table: dict[str, int] = {}

    def ref(text: str) -> int:
        return table.setdefault(text[:max_doc_chars], len(table))

    rows = [
        {"q": g.query[:max_query_chars], "p": [ref(x) for x in g.positives[:max_positives]],
         "n": [ref(x) for x in g.negatives[:max_negatives]], "s": g.source, "r": g.repo, "l": g.language}
        for g in kept
    ]
    blob = json.dumps({"texts": list(table), "groups": rows}, separators=(",", ":")).encode()
    with open(path, "wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0, compresslevel=9) as f:
        f.write(blob)  # no embedded name or timestamp: identical input gives identical bytes
    return {"groups": len(rows), "texts": len(table), "bytes": Path(path).stat().st_size,
            "by_source": dict(Counter(g.source for g in kept))}


def unpack_groups(path: str | Path) -> list[TrainingGroup]:
    with gzip.open(path, "rb") as f:
        data = json.loads(f.read())
    texts = data["texts"]
    return [
        TrainingGroup(row["q"], [texts[i] for i in row["p"]], [texts[i] for i in row["n"]], row["s"], row["r"], row["l"])
        for row in data["groups"]
    ]
