import json
import random
from collections import Counter

import pytest
import torch
import torch.nn.functional as F

from codeintel.config import TrainingConfig
from codeintel.data.schemas import TrainingPair, write_jsonl
from codeintel.training.datasets import GroupBatchSampler, TrainingGroup, group_pairs, make_batch
from codeintel.training.losses import ContrastiveLoss
from codeintel.training.trainer import Trainer


def pairs(n: int, source: str = "docstring") -> list[TrainingPair]:
    return [
        TrainingPair(query=f"query {i} parse json", positive=f"def f{i}(): return json {i}", task="nl2code", source=source)
        for i in range(n)
    ]


def groups(n: int, source: str = "docstring", k: int = 1) -> list[TrainingGroup]:
    return [
        TrainingGroup(f"query {i} parse json", [f"def f{i}_{j}(): return json {i}" for j in range(k)], [], source)
        for i in range(n)
    ]


def test_group_pairs_merges_positives_and_keeps_negatives_clean():
    ps = [
        TrainingPair(query="q", positive="a", task="issue2code", source="issues", repo="r", negatives=["n1", "b"]),
        TrainingPair(query="q", positive="b", task="issue2code", source="issues", repo="r", negatives=["n1", "n2"]),
        TrainingPair(query="q", positive="a", task="issue2code", source="issues", repo="r"),
        TrainingPair(query="q", positive="c", task="issue2code", source="commits", repo="r"),
    ]
    out = {g.source: g for g in group_pairs(ps)}
    assert out["issues"].positives == ["a", "b"] and out["issues"].negatives == ["n1", "n2"]  # "b" is a positive: not a negative
    assert out["commits"].positives == ["c"]


def test_multi_positive_loss_prefers_all_positives_close():
    torch.manual_seed(0)
    q = F.normalize(torch.randn(3, 16), dim=-1)
    d = F.normalize(torch.randn(7, 16), dim=-1)
    index = [[0, 1], [2], [3, 4, 5]]
    loss_fn = ContrastiveLoss(symmetric=False)
    far, _ = loss_fn(q, d, index, ["a", "b", "c"], [f"d{j}" for j in range(7)])
    d_good = d.clone()
    for i, cols in enumerate(index):
        for c in cols:
            d_good[c] = F.normalize(q[i] + 0.1 * d[c], dim=-1)
    near, _ = loss_fn(q, d_good, index, ["a", "b", "c"], [f"d{j}" for j in range(7)])
    assert near < far
    # Moving only one of a query's two positives close helps less than moving both.
    d_half = d.clone()
    d_half[0] = d_good[0]
    for c in (2, 3, 4, 5):
        d_half[c] = d_good[c]
    half, _ = loss_fn(q, d_half, index, ["a", "b", "c"], [f"d{j}" for j in range(7)])
    assert near < half < far


def test_other_positives_are_not_negatives_and_duplicates_are_masked():
    from codeintel.training.losses import duplicate_doc_mask, positive_mask

    cpu = torch.device("cpu")
    index = [[0, 1], [2]]
    keys = ["p0", "p1", "p2", "p0"]  # doc 3 repeats query 0's first positive without being listed for it
    assert positive_mask(index, 4, cpu).tolist() == [[True, True, False, False], [False, False, True, False]]
    assert duplicate_doc_mask(index, keys, cpu).tolist() == [[False, False, False, True], [False, False, False, False]]
    q = F.normalize(torch.randn(2, 8), dim=-1)
    d = F.normalize(torch.randn(4, 8), dim=-1)
    loss_fn = ContrastiveLoss(symmetric=False)
    masked, _ = loss_fn(q, d, index, ["a", "b"], keys)
    unmasked, _ = loss_fn(q, d, index, ["a", "b"], ["p0", "p1", "p2", "other"])
    assert masked < unmasked  # the duplicate no longer counts against query 0


def test_sampler_follows_shares_and_keeps_batches_single_source_and_duplicate_free():
    gs = groups(40, "a") + groups(40, "b") + groups(40, "c") + [TrainingGroup("query 0 parse json", ["dup"], [], "a")]
    sampler = GroupBatchSampler(gs, batch_size=4, shares={"a": 0.75, "b": 0.25}, seed=1)
    seen = Counter()
    it = iter(sampler)
    for _ in range(400):
        batch = next(it)
        sources = {gs[i].source for i in batch}
        assert len(sources) == 1
        assert len({gs[i].query for i in batch}) == len(batch)
        seen[sources.pop()] += 1
    assert "c" not in seen  # no share, never used
    assert 0.65 < seen["a"] / 400 < 0.85
    assert sampler.steps_per_epoch() == (41 + 40) // 4


def test_sampler_retires_a_source_that_only_has_duplicates_left():
    dup = [TrainingGroup("same query", ["same positive"], [], "a") for _ in range(6)]
    sampler = GroupBatchSampler(dup + groups(12, "b"), batch_size=4, shares={"a": 1.0, "b": 1.0}, seed=0)
    batches = [next(iter(sampler)) for _ in range(1)]
    it = iter(GroupBatchSampler(dup + groups(12, "b"), batch_size=4, shares={"a": 1.0, "b": 1.0}, seed=0))
    sources = {next(it)[0] for _ in range(30)}
    assert all(i >= 6 for i in sources) and batches


def test_make_batch_layout_with_several_positives_and_negatives():
    gs = groups(3, k=2)
    gs[0].negatives = ["def wrong(): pass", "def other(): pass"]
    batch = make_batch(gs, hard_negatives=1, rng=random.Random(0), query_prefix="q: ", max_positives=3)
    assert batch.queries[0].startswith("q: ")
    assert batch.positive_index == [[0, 1], [2, 3], [4, 5]]
    assert len(batch.documents) == 7 and batch.documents[6] in gs[0].negatives
    capped = make_batch(groups(1, k=5), 0, random.Random(0), max_positives=2)
    assert len(capped.positive_index[0]) == 2


def _write_train(tmp_path, n=16):
    path = tmp_path / "train.jsonl"
    write_jsonl(path, pairs(n))
    return path


def test_trainer_runs_and_saves_reloadable_checkpoint(tmp_path, tiny_model_cfg):
    model_yaml = tmp_path / "model.yaml"
    model_yaml.write_text(
        f"name: tiny\nadapter: hf\nrepo_id: local/tiny\nlocal_dir: {tiny_model_cfg.local_dir}\ndevice: cpu\n"
    )
    cfg = TrainingConfig(
        model=str(model_yaml), train_files=[str(_write_train(tmp_path))], output_dir=str(tmp_path / "run"),
        batch_size=4, max_steps=3, learning_rate=1e-3, log_every=1, distill_weight=0.5, interpolate_alpha=0.5,
        matryoshka_dims=[16], source_shares={"docstring": 1.0},
    )
    final = Trainer(cfg).train()
    log = [json.loads(line) for line in open(tmp_path / "run" / "train_log.jsonl")]
    assert [r["step"] for r in log if r["event"] == "train"] == [1, 2, 3]
    assert "distill" in log[0] and "nce@16" in log[0]
    assert (final / "codeintel_model.yaml").exists()
    assert (tmp_path / "run" / "final-wise0.5" / "model.safetensors").exists()


def test_gradcache_matches_direct_gradients(tmp_path, tiny_model_cfg):
    model_yaml = tmp_path / "model.yaml"
    model_yaml.write_text(
        f"name: tiny\nadapter: hf\nrepo_id: local/tiny\nlocal_dir: {tiny_model_cfg.local_dir}\ndevice: cpu\n"
    )
    cfg = TrainingConfig(model=str(model_yaml), train_files=[str(_write_train(tmp_path))], output_dir=str(tmp_path / "run"), batch_size=8, source_shares={"docstring": 1.0})
    trainer = Trainer(cfg)
    trainer.student.model.train()
    batch = make_batch(groups(8, k=2), 0, random.Random(0))

    def grads():
        return [p.grad.detach().clone() for p in trainer.student.model.parameters() if p.grad is not None]

    trainer.optimizer.zero_grad()
    direct_logs = trainer._step_direct(batch)
    direct = grads()
    trainer.optimizer.zero_grad()
    cached_logs = trainer._step_gradcache(batch, mbs=3)
    cached = grads()
    assert direct_logs["loss"] == pytest.approx(cached_logs["loss"], rel=1e-5)
    for a, b in zip(direct, cached):
        torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-5)  # fp32 summation order differs between the two paths



def test_hard_negative_mining_skips_positives_of_the_same_query(hashing_model):
    from codeintel.training.datasets import mine_hard_negatives

    q = "fix retry handling"
    recs = [
        TrainingPair(query=q, positive="def retry_once(): return retry(1)", task="issue2code", source="c"),
        TrainingPair(query=q, positive="def retry_twice(): return retry(2)", task="issue2code", source="c"),
        TrainingPair(query="parse json", positive="def parse(): return json", task="nl2code", source="d"),
    ]
    mined = mine_hard_negatives(recs, hashing_model, num_negatives=3, margin=10.0)
    assert "def retry_twice(): return retry(2)" not in mined[0].negatives
    assert mined[0].negatives == ["def parse(): return json"]


def test_mined_negatives_are_unchanged_lexical_neighbours_and_siblings():
    import random

    from codeintel.indexing.storage import build_index
    from codeintel.retrieval.bm25 import BM25Index
    from codeintel.training.negatives import mine_group_negatives, positive_paths

    from .conftest import SAMPLE_REPO

    index = build_index(SAMPLE_REPO)
    positive = next(c.text for c in index.chunks if c.symbol == "apply_discount")
    assert positive_paths([positive]) == {"inventory/pricing.py"}
    negs = mine_group_negatives(index.chunks, BM25Index([c.text for c in index.chunks]), "discount percent price", [positive],
                                {positive}, random.Random(0))
    assert positive not in negs and len(negs) == len(set(negs)) and 3 <= len(negs) <= 8
    texts = {c.text: c for c in index.chunks}
    assert all(texts[n].kind in ("function", "method", "class", "interface", "type", "enum", "variable") for n in negs)
    siblings = {c.text for c in index.chunks if c.path == "inventory/pricing.py" and c.symbol != "apply_discount" and c.kind == "function"}
    assert siblings & set(negs)  # same-file functions (sales_tax, bulk_price) are included


def test_pack_roundtrip_dedupes_truncates_and_caps(tmp_path):
    from codeintel.training.pack import pack_groups, unpack_groups

    shared = "def shared(): pass\n" + "x = 1\n" * 500
    gs = [TrainingGroup(f"q{i}", [shared, f"def p{i}(): pass"], [shared + str(i)[:0], "def n(): pass"], "issues", "r", "python") for i in range(30)]
    gs += [TrainingGroup(f"c{i}", [f"def c{i}(): pass"], [], "calls", "r", "python") for i in range(50)]
    stats = pack_groups(gs, tmp_path / "t.json.gz", max_doc_chars=100, per_source_cap={"calls": 10})
    assert stats["by_source"] == {"issues": 30, "calls": 10} and stats["texts"] < 30 + 10 + 4
    back = unpack_groups(tmp_path / "t.json.gz")
    assert len(back) == 40 and back[0].positives[0] == shared[:100] and back[0].negatives[-1] == "def n(): pass"
    pack_groups(gs, tmp_path / "u.json.gz", max_doc_chars=100, per_source_cap={"calls": 10})
    assert (tmp_path / "t.json.gz").read_bytes() == (tmp_path / "u.json.gz").read_bytes()  # deterministic


def test_teacher_cache_reproduces_live_teacher_distillation(tmp_path, tiny_model_cfg):
    from codeintel.models.base import load_model
    from codeintel.training.pack import pack_groups
    from codeintel.training.teacher import TeacherCache, build_teacher_cache

    model_yaml = tmp_path / "model.yaml"
    model_yaml.write_text(
        f"name: tiny\nadapter: hf\nrepo_id: local/tiny\nlocal_dir: {tiny_model_cfg.local_dir}\ndevice: cpu\n"
    )
    gs = groups(8, k=2)
    packed = tmp_path / "groups.json.gz"
    pack_groups(gs, packed)
    teacher = load_model(tiny_model_cfg)
    cache = build_teacher_cache(teacher, [g.query for g in gs], [t for g in gs for t in (*g.positives, *g.negatives)])
    cache.save(tmp_path / "teacher.pt")
    common = dict(model=str(model_yaml), packed_groups=str(packed), output_dir=str(tmp_path / "run"), batch_size=8,
                  distill_weight=0.5, source_shares={"docstring": 1.0})
    live = Trainer(TrainingConfig(**common))
    cached = Trainer(TrainingConfig(**common, teacher_cache=str(tmp_path / "teacher.pt")))
    assert cached.teacher is None and isinstance(cached.teacher_cache, TeacherCache)
    batch = make_batch(gs, 0, random.Random(0), live.student.query_prefix, live.student.document_prefix)
    tq, td = live._teacher_embeddings(batch)
    cq, cd = cached._teacher_embeddings(batch)
    torch.testing.assert_close(cq, tq.float(), atol=2e-3, rtol=0)  # fp16 storage
    torch.testing.assert_close(cd, td.float(), atol=2e-3, rtol=0)
    with pytest.raises(KeyError):
        cached.teacher_cache.queries(["not in the cache"], "cpu")


def test_eval_every_zero_skips_all_in_training_evaluation(tmp_path, tiny_model_cfg, monkeypatch):
    from codeintel.config import TaskConfig

    model_yaml = tmp_path / "model.yaml"
    model_yaml.write_text(
        f"name: tiny\nadapter: hf\nrepo_id: local/tiny\nlocal_dir: {tiny_model_cfg.local_dir}\ndevice: cpu\n"
    )
    cfg = TrainingConfig(model=str(model_yaml), train_files=[str(_write_train(tmp_path))], output_dir=str(tmp_path / "run"),
                         batch_size=4, max_steps=2, eval_every=0, source_shares={"docstring": 1.0},
                         dev_task=TaskConfig(name="dev", type="localization", path="unused.jsonl"))
    trainer = Trainer(cfg)
    called = []
    monkeypatch.setattr(trainer, "evaluate", lambda: called.append(1))
    trainer.train()
    assert called == []
