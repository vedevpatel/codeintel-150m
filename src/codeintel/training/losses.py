"""Contrastive objectives for bi-encoder training."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F


def duplicate_mask(row_keys: Sequence[str], col_keys: Sequence[str], device: torch.device) -> torch.Tensor:
    """(rows, cols) bool mask, True where col j has the same normalised text as row i's target but j != i.

    Those columns are identical to the label, so treating them as negatives would penalise the right answer.
    """
    mask = torch.zeros(len(row_keys), len(col_keys), dtype=torch.bool)
    positions: dict[str, list[int]] = {}
    for j, k in enumerate(col_keys):
        positions.setdefault(k, []).append(j)
    for i, k in enumerate(row_keys):
        for j in positions.get(k, ()):
            if j != i:
                mask[i, j] = True
    return mask.to(device)


def info_nce(
    q: torch.Tensor, d: torch.Tensor, temperature: float, mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """InfoNCE with row i's label at column i. `d` may carry extra (hard-negative) columns. Returns (loss, logits)."""
    logits = q @ d.T / temperature
    if mask is not None:
        logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    labels = torch.arange(q.size(0), device=q.device)
    return F.cross_entropy(logits, labels), logits


@dataclass
class ContrastiveLoss:
    temperature: float = 0.05
    symmetric: bool = True
    matryoshka_dims: Sequence[int] = ()
    distill_weight: float = 0.0
    distill_temperature: float = 0.05

    def _single(
        self,
        q: torch.Tensor,
        d: torch.Tensor,
        fn_mask: torch.Tensor,
        q_mask: torch.Tensor,
        teacher_sims: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        loss, _ = info_nce(q, d, self.temperature, fn_mask)
        logs = {"nce": float(loss.detach())}
        if self.symmetric:
            b = q.size(0)
            rev, _ = info_nce(d[:b], q, self.temperature, q_mask)
            logs["nce_rev"] = float(rev.detach())
            loss = (loss + rev) / 2
        if teacher_sims is not None and self.distill_weight > 0:
            # Keep the student's ranking distribution close to the frozen base model's (anti-forgetting).
            # Masked columns get a large finite negative so that 0 * log-prob stays finite.
            student = (q @ d.T / self.distill_temperature).masked_fill(fn_mask, -1e4).log_softmax(-1)
            teacher = (teacher_sims / self.distill_temperature).masked_fill(fn_mask, -1e4).softmax(-1)
            kl = F.kl_div(student, teacher, reduction="batchmean")
            logs["distill"] = float(kl.detach())
            loss = loss + self.distill_weight * kl
        return loss, logs

    def __call__(
        self,
        q: torch.Tensor,
        d: torch.Tensor,
        positive_keys: Sequence[str],
        query_keys: Sequence[str],
        document_keys: Sequence[str],
        teacher_q: torch.Tensor | None = None,
        teacher_d: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        q, d = q.float(), d.float()
        fn_mask = duplicate_mask(positive_keys, document_keys, q.device)
        q_mask = duplicate_mask(query_keys, query_keys, q.device)
        teacher_sims = None
        if teacher_q is not None and teacher_d is not None:
            teacher_sims = teacher_q.float() @ teacher_d.float().T
        loss, logs = self._single(q, d, fn_mask, q_mask, teacher_sims)
        dims = [m for m in self.matryoshka_dims if m < q.size(1)]
        for dim in dims:
            qd = F.normalize(q[:, :dim], dim=-1)
            dd = F.normalize(d[:, :dim], dim=-1)
            extra, _ = self._single(qd, dd, fn_mask, q_mask, None)
            logs[f"nce@{dim}"] = float(extra.detach())
            loss = loss + extra
        if dims:
            loss = loss / (1 + len(dims))
        logs["loss"] = float(loss.detach())
        return loss, logs
