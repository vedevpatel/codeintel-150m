"""Contrastive fine-tuning loop for a bi-encoder adapter.

Features: source-homogeneous batches, hard negatives, symmetric InfoNCE with duplicate masking, optional
Matryoshka dims, optional distillation from the frozen base model, GradCache for large effective batches
on small devices, periodic dev evaluation with best-checkpoint selection, and WiSE-FT interpolation.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import random
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from codeintel.config import TrainingConfig, load_model_config
from codeintel.models.base import TransformerEmbedder, load_model
from codeintel.training.datasets import Batch, SourceBatchSampler, load_training_pairs, make_batch
from codeintel.training.losses import ContrastiveLoss

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def autocast_context(precision: str, device: torch.device):
    if precision in ("auto", "fp32"):
        if precision == "auto" and device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[precision]
    return torch.autocast(device.type, dtype=dtype)


def enable_gradient_checkpointing(module: torch.nn.Module) -> None:
    try:
        module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return
    except (AttributeError, TypeError, ValueError):
        pass
    # Remote-code models with the pre-v5 `_set_gradient_checkpointing(module, value)` hook.
    for sub in module.modules():
        if hasattr(sub, "gradient_checkpointing"):
            sub.gradient_checkpointing = True


def linear_warmup_decay(optimizer, warmup: int, total: int):
    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return max(0.0, (total - step) / max(1, total - warmup))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def param_groups(model: torch.nn.Module, weight_decay: float):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or "norm" in name.lower() or name.endswith(".bias") else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


class _RngState:
    """Capture/restore RNG so GradCache's second forward pass reproduces the first (same dropout masks)."""

    def __init__(self, device: torch.device):
        self.device = device
        self.cpu = torch.get_rng_state()
        self.dev = None
        if device.type == "cuda":
            self.dev = torch.cuda.get_rng_state(device)
        elif device.type == "mps":
            self.dev = torch.mps.get_rng_state()

    def restore(self) -> None:
        torch.set_rng_state(self.cpu)
        if self.dev is not None:
            if self.device.type == "cuda":
                torch.cuda.set_rng_state(self.dev, self.device)
            else:
                torch.mps.set_rng_state(self.dev)


class Trainer:
    def __init__(self, cfg: TrainingConfig):
        self.cfg = cfg
        set_seed(cfg.seed)
        self.out_dir = Path(cfg.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        base_cfg = load_model_config(cfg.model)
        # Master weights stay in fp32; mixed precision comes from autocast.
        self.model_cfg = replace(base_cfg, dtype="float32")
        self.student: TransformerEmbedder = load_model(self.model_cfg)
        self.device = self.student.device
        self.teacher: TransformerEmbedder | None = None
        if cfg.distill_weight > 0 or cfg.interpolate_alpha < 1.0:
            self.teacher = load_model(self.model_cfg)
            self.teacher.model.requires_grad_(False)
        if cfg.gradient_checkpointing:
            enable_gradient_checkpointing(self.student.model)
        self.loss_fn = ContrastiveLoss(
            temperature=cfg.temperature,
            symmetric=cfg.symmetric_loss,
            matryoshka_dims=cfg.matryoshka_dims,
            distill_weight=cfg.distill_weight,
            distill_temperature=cfg.distill_temperature,
        )
        self.records = load_training_pairs(cfg.train_files)
        self.sampler = SourceBatchSampler(self.records, cfg.batch_size, cfg.source_weights, cfg.seed)
        steps_per_epoch = len(self.sampler)
        if steps_per_epoch == 0:
            raise ValueError("not enough training records for a single batch")
        self.total_steps = cfg.max_steps or cfg.epochs * steps_per_epoch
        self.optimizer = torch.optim.AdamW(param_groups(self.student.model, cfg.weight_decay), lr=cfg.learning_rate)
        self.scheduler = linear_warmup_decay(self.optimizer, int(cfg.warmup_ratio * self.total_steps), self.total_steps)
        self.rng = random.Random(cfg.seed)
        self.log_path = self.out_dir / "train_log.jsonl"
        self.best_score = -math.inf
        self._dev = None

    # -- forward helpers ---------------------------------------------------------------------------
    def _tokenize(self, model: TransformerEmbedder, texts: list[str], max_length: int):
        return model.tokenize(texts, max_length=min(max_length, model.max_length))

    def _embed(self, model: TransformerEmbedder, texts: list[str], max_length: int) -> torch.Tensor:
        tok = self._tokenize(model, texts, max_length)
        with autocast_context(self.cfg.precision, self.device):
            return model.embed(tok["input_ids"], tok["attention_mask"])

    def _teacher_embeddings(self, batch: Batch):
        if self.teacher is None or self.cfg.distill_weight <= 0:
            return None, None
        with torch.no_grad():
            tq = self._embed(self.teacher, batch.queries, self.cfg.max_query_length)
            td = self._embed(self.teacher, batch.documents, self.cfg.max_document_length)
        return tq, td

    def _step_direct(self, batch: Batch) -> dict[str, float]:
        q = self._embed(self.student, batch.queries, self.cfg.max_query_length)
        d = self._embed(self.student, batch.documents, self.cfg.max_document_length)
        tq, td = self._teacher_embeddings(batch)
        loss, logs = self.loss_fn(q, d, batch.positive_keys, batch.query_keys, batch.document_keys, tq, td)
        loss.backward()
        return logs

    def _step_gradcache(self, batch: Batch, mbs: int) -> dict[str, float]:
        """GradCache (Gao et al. 2021): exact large-batch contrastive gradients with sub-batch memory."""

        def chunks(texts):
            return [texts[i : i + mbs] for i in range(0, len(texts), mbs)]

        groups = [(chunks(batch.queries), self.cfg.max_query_length), (chunks(batch.documents), self.cfg.max_document_length)]
        reps, states = [], []
        with torch.no_grad():
            for parts, max_len in groups:
                outs, sts = [], []
                for part in parts:
                    sts.append(_RngState(self.device))
                    outs.append(self._embed(self.student, part, max_len).float())
                reps.append(torch.cat(outs).requires_grad_())
                states.append(sts)
        tq, td = self._teacher_embeddings(batch)
        loss, logs = self.loss_fn(reps[0], reps[1], batch.positive_keys, batch.query_keys, batch.document_keys, tq, td)
        loss.backward()
        for (parts, max_len), rep, sts in zip(groups, reps, states):
            grads = rep.grad.split([len(p) for p in parts])
            for part, grad, st in zip(parts, grads, sts):
                st.restore()
                emb = self._embed(self.student, part, max_len).float()
                emb.backward(grad)
        return logs

    # -- dev evaluation ----------------------------------------------------------------------------
    def evaluate(self) -> float | None:
        task = self.cfg.dev_task
        if task is None:
            return None
        from codeintel.evaluation.runner import (
            TaskRun,
            load_retrieval_task,
            run_localization_task,
            run_retrieval_task,
        )
        from codeintel.indexing.chunker import Chunker

        ks = [1, 5, 10, 20]
        run = TaskRun()
        systems = [("student", self.student, "dense")]
        self.student.model.eval()
        try:
            if task.type == "retrieval":
                if self._dev is None:
                    self._dev = load_retrieval_task(task, self.cfg.seed)
                run_retrieval_task(self._dev, systems, ks, None, run)
                metrics = run.results["doc"]["student"].metrics
                score = metrics["ndcg@10"]
            else:
                from codeintel.data.schemas import LocalizationCase, read_records

                if self._dev is None:
                    self._dev = read_records(task.path, LocalizationCase)
                repos = Path(task.repos_dir or Path(task.path).parent / "repos")
                run_localization_task(self._dev, repos, Chunker(), systems, ks, None, run)
                metrics = run.results["file"]["student"].metrics
                score = metrics["recall@10"]
        finally:
            self.student.model.train()
        self._log({"event": "eval", **{k: v for k, v in metrics.items() if isinstance(v, float)}})
        return score

    # -- main loop ---------------------------------------------------------------------------------
    def _log(self, row: dict) -> None:
        with open(self.log_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    def _batches(self):
        epoch = 0
        while True:
            for idx in self.sampler:
                yield [self.records[i] for i in idx]
            epoch += 1

    def train(self) -> Path:
        cfg = self.cfg
        (self.out_dir / "training_config.json").write_text(json.dumps(asdict(cfg), indent=2))
        self.student.model.train()
        step = 0
        t0 = time.perf_counter()
        for recs in self._batches():
            if step >= self.total_steps:
                break
            batch = make_batch(recs, cfg.hard_negatives, self.rng, self.student.query_prefix, self.student.document_prefix)
            if cfg.mini_batch_size and cfg.mini_batch_size < len(batch.documents):
                logs = self._step_gradcache(batch, cfg.mini_batch_size)
            else:
                logs = self._step_direct(batch)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.student.model.parameters(), cfg.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % cfg.log_every == 0 or step == 1:
                self._log(
                    {
                        "event": "train",
                        "step": step,
                        "lr": self.scheduler.get_last_lr()[0],
                        "grad_norm": float(grad_norm),
                        "source": batch.source,
                        "elapsed": time.perf_counter() - t0,
                        **logs,
                    }
                )
            if cfg.dev_task is not None and step % cfg.eval_every == 0:
                score = self.evaluate()
                if score is not None and score > self.best_score:
                    self.best_score = score
                    self.save(self.out_dir / "best")
        final = self.out_dir / "final"
        self.save(final)
        if cfg.dev_task is not None:
            score = self.evaluate()
            if score is not None and score > self.best_score:
                self.best_score = score
                self.save(self.out_dir / "best")
        if cfg.interpolate_alpha < 1.0:
            self.interpolate(cfg.interpolate_alpha)
            self.save(self.out_dir / f"final-wise{cfg.interpolate_alpha:g}")
        return final

    @torch.no_grad()
    def interpolate(self, alpha: float) -> None:
        """WiSE-FT: theta = alpha * fine-tuned + (1 - alpha) * base, in place on the student."""
        assert self.teacher is not None
        base = self.teacher.model.state_dict()
        for name, tensor in self.student.model.state_dict().items():
            if tensor.is_floating_point():
                tensor.mul_(alpha).add_(base[name].to(tensor.device), alpha=1 - alpha)

    def save(self, directory: Path) -> None:
        self.student.save(directory)
        # A model config that reloads this checkpoint through the generic adapter.
        lines = [
            f"name: {self.model_cfg.name}-ft",
            "adapter: hf",
            f"repo_id: {self.model_cfg.repo_id}",
            f"local_dir: {directory.resolve()}",
            f"trust_remote_code: {str(self.model_cfg.trust_remote_code).lower()}",
        ]
        if self.model_cfg.remote_code_sha256:
            lines.append("remote_code_sha256:")
            lines += [f"  {k}: {v}" for k, v in self.model_cfg.remote_code_sha256.items()]
        if self.model_cfg.max_length:
            lines.append(f"max_length: {self.model_cfg.max_length}")
        (directory / "codeintel_model.yaml").write_text("\n".join(lines) + "\n")


def train(cfg: TrainingConfig) -> Path:
    return Trainer(cfg).train()
