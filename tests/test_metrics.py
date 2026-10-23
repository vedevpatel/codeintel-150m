import math

import pytest

from codeintel.evaluation.metrics import (
    average_precision,
    bootstrap_ci,
    evaluate_rankings,
    full_recall_at_k,
    hit_at_k,
    mrr_at_k,
    ndcg_at_k,
    paired_comparison,
    recall_at_k,
)

RANKED = ["a", "b", "c", "d", "e"]


def test_binary_metrics():
    rel = {"b", "e", "z"}
    assert hit_at_k(RANKED, rel, 1) == 0.0
    assert hit_at_k(RANKED, rel, 2) == 1.0
    assert recall_at_k(RANKED, rel, 5) == pytest.approx(2 / 3)
    assert full_recall_at_k(RANKED, rel, 5) == 0.0
    assert full_recall_at_k(RANKED, {"a", "c"}, 3) == 1.0
    assert mrr_at_k(RANKED, rel) == 0.5
    assert mrr_at_k(RANKED, rel, k=1) == 0.0


def test_ndcg_matches_hand_computation():
    rel = {"b", "e"}
    dcg = 1 / math.log2(3) + 1 / math.log2(6)
    idcg = 1 + 1 / math.log2(3)
    assert ndcg_at_k(RANKED, rel, 5) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(["b", "e"], rel, 2) == pytest.approx(1.0)


def test_graded_ndcg_and_duplicates():
    rel = {"a": 1.0, "b": 2.0}
    assert ndcg_at_k(["b", "a"], rel, 2) == pytest.approx(1.0)
    assert ndcg_at_k(["a", "b"], rel, 2) < 1.0
    # A repeated relevant key must not be double counted.
    assert ndcg_at_k(["b", "b", "a"], rel, 3) <= 1.0


def test_average_precision():
    assert average_precision(RANKED, {"a", "c"}) == pytest.approx((1 + 2 / 3) / 2)


def test_evaluate_rankings_skips_unjudged_and_zero_fills_missing():
    means, per_query = evaluate_rankings({"q1": ["a"]}, {"q1": {"a"}, "q2": {"b"}, "q3": set()}, ks=[1])
    assert means["num_queries"] == 2
    assert per_query["q2"]["hit@1"] == 0.0
    assert means["hit@1"] == 0.5


def test_paired_comparison():
    same = paired_comparison([0.5, 0.2, 0.9], [0.5, 0.2, 0.9])
    assert same["mean_diff"] == 0 and same["p_value"] == 1.0
    better = paired_comparison([1.0] * 40, [0.0] * 40, samples=2000)
    assert better["mean_diff"] == 1.0 and better["p_value"] < 0.01 and better["ci_low"] == 1.0


def test_bootstrap_ci_brackets_mean():
    vals = [0.0, 1.0] * 50
    lo, hi = bootstrap_ci(vals, samples=500)
    assert lo < 0.5 < hi
