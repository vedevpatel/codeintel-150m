import json
import random
from dataclasses import replace

import pytest
import torch

from codeintel.config import TrainingConfig
from codeintel.data.schemas import TrainingPair, write_jsonl
from codeintel.training.datasets import SourceBatchSampler, make_batch
from codeintel.training.losses import ContrastiveLoss, duplicate_mask, info_nce
from codeintel.training.trainer import Trainer


def pairs(n: int, source: str = "docstring") -> list[TrainingPair]:
    return [
        TrainingPair(query=f"query {i} parse json", positive=f"def f{i}(): return json {i}", task="nl2code", source=source)
        for i in range(n)
    ]


def test_duplicate_mask_marks_identical_non_label_columns():
    mask = duplicate_mask(["a", "b"], ["a", "b", "a", "c"], torch.device("cpu"))
    assert mask.tolist() == [[False, False, True, False], [False, False, False, False]]


def test_info_nce_prefers_aligned_pairs():
    q = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    aligned, _ = info_nce(q, q, 0.05)
    shuffled, _ = info_nce(q, q.roll(1, 0), 0.05)
    assert aligned < shuffled


def test_masked_duplicates_do_not_count_as_negatives():
    q = torch.nn.functional.normalize(torch.randn(2, 8), dim=-1)
    d = torch.cat([q, q[:1]])  # third column duplicates row 0's positive
    loss_fn = ContrastiveLoss(symmetric=False)
    masked, _ = loss_fn(q, d, ["p0", "p1"], ["q0", "q1"], ["p0", "p1", "p0"])
    unmasked, _ = info_nce(q, d, 0.05)
    assert masked < unmasked


def test_sampler_batches_are_single_source_and_duplicate_free():
    records = pairs(10, "a") + pairs(10, "b") + [replace(pairs(1, "a")[0])] * 5
    sampler = SourceBatchSampler(records, batch_size=4, seed=1)
    batches = list(sampler)
    assert batches
    for batch in batches:
        assert len({records[i].source for i in batch}) == 1
        assert len({records[i].query for i in batch}) == len(batch)


def test_make_batch_layout():
    recs = pairs(3)
    recs[0] = replace(recs[0], negatives=["def wrong(): pass"])
    batch = make_batch(recs, hard_negatives=1, rng=random.Random(0), query_prefix="q: ")
    assert batch.queries[0].startswith("q: ")
    assert len(batch.documents) == 4 and batch.documents[3] == "def wrong(): pass"


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
        matryoshka_dims=[16],
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
    cfg = TrainingConfig(model=str(model_yaml), train_files=[str(_write_train(tmp_path))], output_dir=str(tmp_path / "run"), batch_size=8)
    trainer = Trainer(cfg)
    trainer.student.model.train()
    batch = make_batch(pairs(8), 0, random.Random(0))

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
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)


def test_sampler_retires_a_source_that_only_has_duplicates_left():
    dup = pairs(1, "a")[0]
    records = [dup] * 6 + pairs(12, "b")
    batches = list(SourceBatchSampler(records, batch_size=4, seed=0))
    assert sum(1 for b in batches if records[b[0]].source == "b") == 3


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
