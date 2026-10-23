"""Exact inner-product search over L2-normalised embeddings."""

from __future__ import annotations

import numpy as np


def top_k(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise top-k of a (q, n) score matrix: descending score, lowest index first among ties.

    A full stable sort (rather than argpartition) keeps the choice among ties at the cutoff deterministic.
    """
    n = scores.shape[1]
    k = min(k, n)
    if k <= 0:
        return np.zeros((scores.shape[0], 0), np.int64), np.zeros((scores.shape[0], 0), np.float32)
    idx = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    return idx, np.take_along_axis(scores, idx, axis=1)


class DenseIndex:
    def __init__(self, embeddings: np.ndarray, query_block: int = 64):
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.query_block = query_block

    def __len__(self) -> int:
        return len(self.embeddings)

    def search(self, queries: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        queries = np.atleast_2d(np.asarray(queries, dtype=np.float32))
        all_idx, all_scores = [], []
        for i in range(0, len(queries), self.query_block):
            scores = queries[i : i + self.query_block] @ self.embeddings.T
            idx, sc = top_k(scores, k)
            all_idx.append(idx)
            all_scores.append(sc)
        if not all_idx:
            return np.zeros((0, 0), np.int64), np.zeros((0, 0), np.float32)
        return np.concatenate(all_idx), np.concatenate(all_scores)
