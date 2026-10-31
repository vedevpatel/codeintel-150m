"""Hard negatives from the repository snapshot a training query was written against.

For a query with known positives (the symbols a fix edited) two kinds of unchanged code make useful negatives:
  * BM25 hits: lexically close to the query but not edited, which is what a retriever confuses with the answer;
  * siblings: other functions in the files the fix touched, which share vocabulary and context with the answer.
Chunks whose text equals any positive of the same snapshot are never negatives. Rendering matches the positives
(same chunker, same header), so a negative differs from a positive only in being unchanged.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence

from codeintel.retrieval.bm25 import BM25Index
from codeintel.schemas import Chunk

NEGATIVE_KINDS = frozenset({"function", "method", "class", "interface", "type", "enum", "variable"})


def positive_paths(positives: Iterable[str]) -> set[str]:
    """File paths from the `# path::Symbol` headers of rendered positives."""
    paths = set()
    for text in positives:
        header = text.split("\n", 1)[0]
        label = header.split(" ", 1)[1] if " " in header else ""
        if "::" in label:
            paths.add(label.split("::", 1)[0])
    return paths


def positive_labels(positives: Iterable[str]) -> set[str]:
    """`path::Symbol` labels from the headers of rendered positives."""
    labels = set()
    for text in positives:
        header = text.split("\n", 1)[0]
        label = header.split(" ", 1)[1] if " " in header else ""
        if "::" in label:
            labels.add(label)
    return labels


def _label(chunk: Chunk) -> str:
    header = chunk.text.split("\n", 1)[0]
    return header.split(" ", 1)[1] if " " in header else ""


def mine_group_negatives(
    chunks: Sequence[Chunk],
    bm25: BM25Index,
    query: str,
    positives: Sequence[str],
    all_positives: set[str],
    rng: random.Random,
    n_bm25: int = 5,
    n_siblings: int = 3,
    depth: int = 40,
    banned_labels: set[str] | None = None,
) -> list[str]:
    """Up to `n_bm25` lexical hits and `n_siblings` same-file chunks that are not positives of this snapshot.

    `banned_labels` (`path::Symbol`) also excludes other versions of the positives, for when the snapshot is not the
    one the positives were taken from (e.g. one HEAD index per repository): the fixed version of an edited function
    must never be a negative.
    """
    banned = set(positives) | all_positives
    labels = banned_labels or set()
    out: list[str] = []
    for idx, _score in bm25.search(query, depth):
        c = chunks[idx]
        if c.kind in NEGATIVE_KINDS and c.text not in banned and c.text not in out and _label(c) not in labels:
            out.append(c.text)
            if len(out) >= n_bm25:
                break
    paths = positive_paths(positives)
    siblings = [c.text for c in chunks if c.path in paths and c.kind in NEGATIVE_KINDS and c.text not in banned
                and c.text not in out and _label(c) not in labels]
    rng.shuffle(siblings)
    return out + siblings[:n_siblings]
