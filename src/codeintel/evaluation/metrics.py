"""Ranking metrics and paired significance tests.

All metrics take a ranked list of keys and a mapping of relevant key -> graded relevance (> 0).
Queries without any relevant key are skipped by `evaluate_rankings` (and counted).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

Relevance = Mapping[str, float]


def _rel(relevant: Relevance | set[str]) -> dict[str, float]:
    if isinstance(relevant, Mapping):
        return {k: float(v) for k, v in relevant.items() if v > 0}
    return dict.fromkeys(relevant, 1.0)


def hit_at_k(ranked: Sequence[str], relevant: Relevance | set[str], k: int) -> float:
    rel = _rel(relevant)
    return float(any(doc in rel for doc in ranked[:k]))


def recall_at_k(ranked: Sequence[str], relevant: Relevance | set[str], k: int) -> float:
    rel = _rel(relevant)
    if not rel:
        return 0.0
    return len(set(ranked[:k]) & rel.keys()) / len(rel)


def full_recall_at_k(ranked: Sequence[str], relevant: Relevance | set[str], k: int) -> float:
    """1 if *every* relevant key is in the top k (SWE-bench-style localization accuracy)."""
    rel = _rel(relevant)
    return float(bool(rel) and rel.keys() <= set(ranked[:k]))


def precision_at_k(ranked: Sequence[str], relevant: Relevance | set[str], k: int) -> float:
    rel = _rel(relevant)
    return sum(1 for doc in ranked[:k] if doc in rel) / k if k else 0.0


def mrr_at_k(ranked: Sequence[str], relevant: Relevance | set[str], k: int | None = None) -> float:
    rel = _rel(relevant)
    for i, doc in enumerate(ranked[:k] if k else ranked):
        if doc in rel:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: Relevance | set[str], k: int) -> float:
    rel = _rel(relevant)
    seen: set[str] = set()
    dcg = 0.0
    for i, doc in enumerate(ranked[:k]):
        if doc in rel and doc not in seen:
            dcg += (2 ** rel[doc] - 1) / math.log2(i + 2)
            seen.add(doc)
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision(ranked: Sequence[str], relevant: Relevance | set[str], k: int | None = None) -> float:
    rel = _rel(relevant)
    if not rel:
        return 0.0
    hits, total = 0, 0.0
    for i, doc in enumerate(ranked[:k] if k else ranked):
        if doc in rel:
            hits += 1
            total += hits / (i + 1)
    return total / len(rel)


def query_metrics(ranked: Sequence[str], relevant: Relevance | set[str], ks: Sequence[int]) -> dict[str, float]:
    out: dict[str, float] = {"mrr": mrr_at_k(ranked, relevant)}
    for k in ks:
        out[f"ndcg@{k}"] = ndcg_at_k(ranked, relevant, k)
        out[f"recall@{k}"] = recall_at_k(ranked, relevant, k)
        out[f"hit@{k}"] = hit_at_k(ranked, relevant, k)
        out[f"full_recall@{k}"] = full_recall_at_k(ranked, relevant, k)
    out[f"map@{max(ks)}"] = average_precision(ranked, relevant, max(ks))
    return out


def evaluate_rankings(
    rankings: Mapping[str, Sequence[str]], qrels: Mapping[str, Relevance | set[str]], ks: Sequence[int]
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Mean metrics and per-query metrics. Queries absent from `rankings` score zero."""
    per_query: dict[str, dict[str, float]] = {}
    for qid, relevant in qrels.items():
        if not _rel(relevant):
            continue
        per_query[qid] = query_metrics(list(rankings.get(qid, [])), relevant, ks)
    if not per_query:
        return {"num_queries": 0}, {}
    names = next(iter(per_query.values())).keys()
    means = {m: float(np.mean([pq[m] for pq in per_query.values()])) for m in names}
    means["num_queries"] = len(per_query)
    return means, per_query


def bootstrap_ci(values: Sequence[float], samples: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, len(arr), size=(samples, len(arr)))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_comparison(
    a: Sequence[float], b: Sequence[float], samples: int = 10000, seed: int = 0, alpha: float = 0.05
) -> dict[str, float]:
    """Compare per-query scores of system A vs B on the same queries.

    Returns the mean difference (A - B), a bootstrap CI of that difference, and a two-sided p-value from a
    paired sign-flip randomization test.
    """
    x = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if len(x) == 0:
        return {"mean_diff": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0, "n": 0}
    rng = np.random.default_rng(seed)
    boot = x[rng.integers(0, len(x), size=(samples, len(x)))].mean(axis=1)
    observed = abs(x.mean())
    signs = rng.choice([-1.0, 1.0], size=(samples, len(x)))
    flipped = np.abs((signs * x).mean(axis=1))
    p = (np.sum(flipped >= observed - 1e-12) + 1) / (samples + 1)
    return {
        "mean_diff": float(x.mean()),
        "ci_low": float(np.quantile(boot, alpha / 2)),
        "ci_high": float(np.quantile(boot, 1 - alpha / 2)),
        "p_value": float(p),
        "n": int(len(x)),
    }
