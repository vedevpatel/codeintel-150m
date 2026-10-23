"""Training data loading, source-homogeneous batch sampling, and hard-negative mining."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from codeintel.data.schemas import TrainingPair, read_records
from codeintel.evaluation.contamination import exact_key
from codeintel.retrieval.dense import DenseIndex

if TYPE_CHECKING:
    from codeintel.models.base import EmbeddingModel


def load_training_pairs(paths: Sequence[str]) -> list[TrainingPair]:
    records: list[TrainingPair] = []
    for path in paths:
        records.extend(read_records(path, TrainingPair))
    return records


@dataclass
class Batch:
    queries: list[str]
    documents: list[str]  # positives first (row-aligned with queries), then hard negatives
    positive_keys: list[str]  # normalised-text key of each query's positive
    query_keys: list[str]
    document_keys: list[str]
    source: str


class SourceBatchSampler:
    """Yield batches drawn from a single source at a time.

    Source-homogeneous batches keep in-batch negatives meaningful (an issue report is not contrasted against
    docstring pairs from an unrelated task). Sources are sampled in proportion to weight * size. Records
    whose query or positive duplicates one already in the batch are deferred, which removes the most common
    in-batch false negatives.
    """

    def __init__(
        self,
        records: Sequence[TrainingPair],
        batch_size: int,
        weights: Mapping[str, float] | None = None,
        seed: int = 13,
    ):
        if batch_size < 2:
            raise ValueError("contrastive training needs batch_size >= 2")
        self.records = records
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        self.by_source: dict[str, list[int]] = defaultdict(list)
        for i, r in enumerate(records):
            self.by_source[r.source].append(i)
        weights = weights or {}
        self.weights = {s: weights.get(s, 1.0) * len(ix) for s, ix in self.by_source.items()}
        self.query_keys = [exact_key(r.query) for r in records]
        self.pos_keys = [exact_key(r.positive) for r in records]

    def __len__(self) -> int:
        return sum(len(ix) // self.batch_size for ix in self.by_source.values())

    def __iter__(self) -> Iterator[list[int]]:
        pools = {s: self.rng.sample(ix, len(ix)) for s, ix in self.by_source.items()}
        while True:
            live = [s for s, pool in pools.items() if len(pool) >= self.batch_size and self.weights[s] > 0]
            if not live:
                return
            source = self.rng.choices(live, weights=[self.weights[s] for s in live])[0]
            pool = pools[source]
            before = len(pool)
            batch, deferred, seen = [], [], set()
            while pool and len(batch) < self.batch_size:
                i = pool.pop()
                keys = (self.query_keys[i], self.pos_keys[i])
                if keys[0] in seen or keys[1] in seen:
                    deferred.append(i)
                    continue
                seen.update(keys)
                batch.append(i)
            pool.extend(deferred)
            if len(batch) < self.batch_size:
                # Only duplicates are left in this source; retire it and keep sampling the others.
                pool.extend(batch)
                assert len(pool) == before
                pools[source] = []
                continue
            yield batch


def make_batch(
    records: Sequence[TrainingPair], hard_negatives: int, rng: random.Random, query_prefix: str = "", document_prefix: str = ""
) -> Batch:
    queries = [query_prefix + r.query for r in records]
    positives = [document_prefix + r.positive for r in records]
    negatives: list[str] = []
    for r in records:
        if hard_negatives and r.negatives:
            negatives.extend(document_prefix + n for n in rng.sample(r.negatives, min(hard_negatives, len(r.negatives))))
    documents = positives + negatives
    return Batch(
        queries=queries,
        documents=documents,
        positive_keys=[exact_key(r.positive) for r in records],
        query_keys=[exact_key(r.query) for r in records],
        document_keys=[exact_key(d[len(document_prefix):]) for d in documents],
        source=records[0].source if records else "",
    )


def mine_hard_negatives(
    records: Sequence[TrainingPair],
    model: EmbeddingModel,
    num_negatives: int = 3,
    depth: int = 50,
    margin: float = 0.95,
    corpus: Sequence[str] | None = None,
) -> list[TrainingPair]:
    """Attach dense-retrieved hard negatives to each record.

    Candidates come from `corpus` (default: all positives in `records`). A candidate is rejected if it is the
    record's own positive or scores above `margin` x the positive's score -- such candidates are frequently
    unlabeled positives (TopK-PercPos filtering, as in NV-Retriever). Positives of *any* record with the same
    query are excluded as well.
    """
    corpus = list(dict.fromkeys(corpus if corpus is not None else (r.positive for r in records)))
    corpus_keys = [exact_key(t) for t in corpus]
    doc_emb = model.encode_documents(corpus)
    q_emb = model.encode_queries([r.query for r in records])
    pos_emb = model.encode_documents([r.positive for r in records])
    pos_scores = np.einsum("ij,ij->i", q_emb, pos_emb)
    idx, scores = DenseIndex(doc_emb).search(q_emb, depth)
    # A query can have several positives (e.g. every function a commit changed); none may become a negative.
    positives_of: dict[str, set[str]] = defaultdict(set)
    for r in records:
        positives_of[exact_key(r.query)].add(exact_key(r.positive))
    out = []
    for r, row_i, row_s, ps in zip(records, idx, scores, pos_scores):
        own = positives_of[exact_key(r.query)]
        picked = []
        for j, s in zip(row_i, row_s):
            if corpus_keys[j] in own or s > margin * ps:
                continue
            picked.append(corpus[j])
            if len(picked) == num_negatives:
                break
        out.append(replace(r, negatives=picked))
    return out
