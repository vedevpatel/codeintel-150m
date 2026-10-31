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
class TrainingGroup:
    """One query with all of its positives (e.g. every function a fix edited) and its hard negatives."""

    query: str
    positives: list[str]
    negatives: list[str]
    source: str
    repo: str = ""
    language: str = ""


def group_pairs(pairs: Sequence[TrainingPair], clean_sources: Sequence[str] = ("issues",)) -> list[TrainingGroup]:
    """Merge pairs that share (source, repo, query) into groups; negatives never include a group's own positives.

    Queries from `clean_sources` (real issue text) go through the same cleaning as evaluation queries, so the
    model is trained on what it is later asked.
    """
    from codeintel.retrieval.query import clean_issue_query

    merged: dict[tuple[str, str, str], TrainingGroup] = {}
    for p in pairs:
        g = merged.setdefault((p.source, p.repo, p.query), TrainingGroup(p.query, [], [], p.source, p.repo, p.language))
        if p.positive not in g.positives:
            g.positives.append(p.positive)
        for n in p.negatives:
            if n not in g.negatives:
                g.negatives.append(n)
    groups = list(merged.values())
    for g in groups:
        own = set(g.positives)
        g.negatives = [n for n in g.negatives if n not in own]
        if g.source in clean_sources:
            g.query = clean_issue_query(g.query)
    return groups


@dataclass
class Batch:
    queries: list[str]
    documents: list[str]  # all positives (grouped by query, in query order), then hard negatives
    positive_index: list[list[int]]  # per query: indices into `documents` of its positives
    query_keys: list[str]
    document_keys: list[str]
    source: str


class GroupBatchSampler:
    """Infinite stream of batches of group indices, each batch drawn from a single source.

    Source-homogeneous batches keep in-batch negatives meaningful (an issue report is not contrasted against
    docstring pairs from an unrelated task). A batch's source is drawn with probability proportional to
    `shares[source]` (sources without a share get 0 and are never used), so the training mixture is specified
    directly, e.g. 20% issue reports. Groups whose query, or any positive, duplicates one already in the batch
    are deferred, which removes the most common in-batch false negatives.
    """

    def __init__(self, groups: Sequence[TrainingGroup], batch_size: int, shares: Mapping[str, float], seed: int = 13):
        if batch_size < 2:
            raise ValueError("contrastive training needs batch_size >= 2")
        self.groups, self.batch_size, self.rng = groups, batch_size, random.Random(seed)
        self.by_source: dict[str, list[int]] = defaultdict(list)
        for i, g in enumerate(groups):
            self.by_source[g.source].append(i)
        self.shares = {s: shares.get(s, 0.0) for s in self.by_source if shares.get(s, 0.0) > 0}
        usable = [s for s in self.shares if len(self.by_source[s]) >= batch_size]
        if not usable:
            raise ValueError(f"no source with a positive share has {batch_size} groups; sources: "
                             f"{ {s: len(v) for s, v in self.by_source.items()} }, shares: {dict(shares)}")
        self.shares = {s: self.shares[s] for s in usable}
        self.query_keys = [exact_key(g.query) for g in groups]
        self.pos_keys = [{exact_key(p) for p in g.positives} for g in groups]
        self._pools: dict[str, list[int]] = {s: [] for s in self.shares}

    def _refill(self, source: str) -> None:
        self._pools[source] = self.rng.sample(self.by_source[source], len(self.by_source[source]))

    def steps_per_epoch(self) -> int:
        """Batches needed to pass over every group of the used sources once."""
        return sum(len(self.by_source[s]) for s in self.shares) // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        while True:
            source = self.rng.choices(list(self.shares), weights=list(self.shares.values()))[0]
            batch, seen_q, seen_p, deferred, scans = [], set(), set(), [], 0
            while len(batch) < self.batch_size:
                if not self._pools[source]:
                    self._refill(source)
                    scans += 1
                    if scans > 3:  # fewer distinct groups than the batch size after de-duplication
                        break
                i = self._pools[source].pop()
                if self.query_keys[i] in seen_q or (self.pos_keys[i] & seen_p):
                    deferred.append(i)
                    continue
                seen_q.add(self.query_keys[i])
                seen_p |= self.pos_keys[i]
                batch.append(i)
            self._pools[source].extend(deferred)
            if len(batch) == self.batch_size:
                yield batch
            else:  # could not fill a duplicate-free batch from this source: drop it for the rest of the run
                self.shares.pop(source)
                if not self.shares:
                    return


def make_batch(
    groups: Sequence[TrainingGroup], hard_negatives: int, rng: random.Random, query_prefix: str = "",
    document_prefix: str = "", max_positives: int = 3,
) -> Batch:
    """Queries, then every query's positives (at most `max_positives`, sampled), then sampled hard negatives."""
    queries = [query_prefix + g.query for g in groups]
    documents: list[str] = []
    positive_index: list[list[int]] = []
    for g in groups:
        chosen = g.positives if len(g.positives) <= max_positives else rng.sample(g.positives, max_positives)
        positive_index.append(list(range(len(documents), len(documents) + len(chosen))))
        documents.extend(document_prefix + p for p in chosen)
    for g in groups:
        if hard_negatives and g.negatives:
            documents.extend(document_prefix + n for n in rng.sample(g.negatives, min(hard_negatives, len(g.negatives))))
    return Batch(
        queries=queries,
        documents=documents,
        positive_index=positive_index,
        query_keys=[exact_key(g.query) for g in groups],
        document_keys=[exact_key(d[len(document_prefix):]) for d in documents],
        source=groups[0].source if groups else "",
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
