"""Dense + lexical retrieval over a CodeIndex, rank fusion, and chunk -> file/symbol aggregation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np

from codeintel.retrieval.bm25 import BM25Index
from codeintel.retrieval.dense import DenseIndex
from codeintel.schemas import Chunk, SearchHit

if TYPE_CHECKING:
    from codeintel.indexing.storage import CodeIndex
    from codeintel.models.base import EmbeddingModel

MODES = ("dense", "bm25", "hybrid")


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]], k: int = 60, weights: Sequence[float] | None = None
) -> list[tuple[int, float]]:
    weights = weights or [1.0] * len(rankings)
    fused: dict[int, float] = defaultdict(float)
    for ranking, w in zip(rankings, weights):
        for rank, doc in enumerate(ranking):
            fused[doc] += w / (k + rank + 1)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))


def file_key(chunk: Chunk) -> str:
    return chunk.path


def symbol_key(chunk: Chunk) -> str:
    return f"{chunk.path}::{chunk.symbol}" if chunk.symbol else f"{chunk.path}::<module>"


def aggregate(
    hits: Sequence[tuple[int, float]], chunks: Sequence[Chunk], key: Callable[[Chunk], str], k: int | None = None
) -> list[SearchHit]:
    """Collapse ranked chunk hits to unique keys (files, symbols) by max score, keeping first-seen order."""
    best: dict[str, tuple[float, int]] = {}
    for order, (idx, score) in enumerate(hits):
        kk = key(chunks[idx])
        if kk not in best:
            best[kk] = (score, order)
    ranked = sorted(best.items(), key=lambda kv: (kv[1][1]))
    ranked = ranked[:k] if k else ranked
    return [SearchHit(key=kk, score=float(s), rank=r + 1) for r, (kk, (s, _)) in enumerate(ranked)]


class Retriever:
    """Search a CodeIndex with dense, BM25, or hybrid (RRF) ranking."""

    def __init__(
        self,
        index: CodeIndex,
        model: EmbeddingModel | None = None,
        rrf_k: int = 60,
        candidates: int = 200,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
        bm25: BM25Index | None = None,
    ):
        self.index = index
        self.model = model
        self.rrf_k, self.candidates = rrf_k, candidates
        self.dense_weight, self.bm25_weight = dense_weight, bm25_weight
        self._bm25 = bm25
        self.dense = DenseIndex(index.embeddings) if index.embeddings is not None else None

    @property
    def bm25(self) -> BM25Index:
        if self._bm25 is None:
            self._bm25 = BM25Index([c.text for c in self.index.chunks])
        return self._bm25

    def _dense_hits(self, query_vecs: np.ndarray, k: int) -> list[list[tuple[int, float]]]:
        if self.dense is None:
            raise RuntimeError("index has no embeddings; rebuild it with a model or use mode='bm25'")
        idx, scores = self.dense.search(query_vecs, k)
        return [[(int(i), float(s)) for i, s in zip(row_i, row_s)] for row_i, row_s in zip(idx, scores)]

    def search_many(self, queries: Sequence[str], k: int = 10, mode: str = "hybrid") -> list[list[tuple[int, float]]]:
        """Ranked (chunk_index, score) lists, one per query."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        depth = max(k, self.candidates)
        dense_hits = None
        if mode in ("dense", "hybrid"):
            if self.model is None:
                raise RuntimeError("dense retrieval requires a model")
            dense_hits = self._dense_hits(self.model.encode_queries(list(queries)), depth)
            if mode == "dense":
                return [h[:k] for h in dense_hits]
        lexical = [self.bm25.search(q, depth) for q in queries]
        if mode == "bm25":
            return [h[:k] for h in lexical]
        assert dense_hits is not None
        return [
            reciprocal_rank_fusion(
                [[i for i, _ in d], [i for i, _ in b]], k=self.rrf_k, weights=[self.dense_weight, self.bm25_weight]
            )[:k]
            for d, b in zip(dense_hits, lexical)
        ]

    def search(self, query: str, k: int = 10, mode: str = "hybrid") -> list[tuple[Chunk, float]]:
        return [(self.index.chunks[i], s) for i, s in self.search_many([query], k, mode)[0]]
