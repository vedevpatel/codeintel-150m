"""Contrastive objectives for bi-encoder training.

A fix usually edits several functions, so a query has a *set* of positives. The loss is the average, over a
query's positives, of an InfoNCE term whose denominator holds that positive and every true negative (in-batch
positives of other queries and mined hard negatives), but not the query's other positives. Documents whose text
duplicates one of a query's positives without being listed for it are masked out of its denominator.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

NEG = -1e9  # finite "minus infinity": keeps gradients defined for rows where everything is masked


def positive_mask(positive_index: Sequence[Sequence[int]], n_docs: int, device: torch.device) -> torch.Tensor:
    """(queries, docs) bool matrix, True where the document is a positive of the query."""
    mask = torch.zeros(len(positive_index), n_docs, dtype=torch.bool)
    for i, cols in enumerate(positive_index):
        mask[i, list(cols)] = True
    return mask.to(device)


def duplicate_doc_mask(positive_index: Sequence[Sequence[int]], document_keys: Sequence[str], device: torch.device) -> torch.Tensor:
    """True where a document has the same normalised text as one of the query's positives but is not listed
    as a positive of that query: treating it as a negative would penalise the right answer."""
    keys_at: dict[str, list[int]] = {}
    for j, k in enumerate(document_keys):
        keys_at.setdefault(k, []).append(j)
    mask = torch.zeros(len(positive_index), len(document_keys), dtype=torch.bool)
    for i, cols in enumerate(positive_index):
        own = set(cols)
        for c in cols:
            for j in keys_at[document_keys[c]]:
                if j not in own:
                    mask[i, j] = True
    return mask.to(device)


def duplicate_query_mask(query_keys: Sequence[str], device: torch.device) -> torch.Tensor:
    """(queries, queries) True where two different queries have identical normalised text."""
    keys = list(query_keys)
    mask = torch.tensor([[a == b and i != j for j, b in enumerate(keys)] for i, a in enumerate(keys)], dtype=torch.bool)
    return mask.to(device)


def multi_positive_nce(
    q: torch.Tensor, d: torch.Tensor, pos: torch.Tensor, dup: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Mean over queries of the mean over their positives of -log(exp(s_p) / (exp(s_p) + sum_neg exp(s_n)))."""
    logits = q @ d.T / temperature
    negatives = logits.masked_fill(pos | dup, NEG)
    neg_lse = torch.logsumexp(negatives, dim=1, keepdim=True)  # (B, 1)
    per_positive = F.softplus(neg_lse - logits)  # == -log(exp(l) / (exp(l) + exp(neg_lse)))
    per_query = (per_positive * pos).sum(1) / pos.sum(1).clamp(min=1)
    return per_query[pos.any(1)].mean()


def reverse_nce(q: torch.Tensor, d: torch.Tensor, pos: torch.Tensor, qdup: torch.Tensor, temperature: float) -> torch.Tensor:
    """Document -> query direction for every positive document, labelled with its (first) owning query."""
    owned = pos.any(0)
    owner = pos.float().argmax(0)[owned]
    logits = d[owned] @ q.T / temperature
    logits = logits.masked_fill(qdup[owner], NEG)
    return F.cross_entropy(logits, owner)


@dataclass
class ContrastiveLoss:
    temperature: float = 0.05
    symmetric: bool = True
    matryoshka_dims: Sequence[int] = ()
    distill_weight: float = 0.0
    distill_temperature: float = 0.05

    def _single(self, q, d, pos, dup, qdup, teacher_sims) -> tuple[torch.Tensor, dict[str, float]]:
        loss = multi_positive_nce(q, d, pos, dup, self.temperature)
        logs = {"nce": float(loss.detach())}
        if self.symmetric:
            rev = reverse_nce(q, d, pos, qdup, self.temperature)
            logs["nce_rev"] = float(rev.detach())
            loss = (loss + rev) / 2
        if teacher_sims is not None and self.distill_weight > 0:
            # Keep the student's ranking distribution close to the frozen base model's (anti-forgetting).
            student = (q @ d.T / self.distill_temperature).masked_fill(dup, NEG).log_softmax(-1)
            teacher = (teacher_sims / self.distill_temperature).masked_fill(dup, NEG).softmax(-1)
            kl = F.kl_div(student, teacher, reduction="batchmean")
            logs["distill"] = float(kl.detach())
            loss = loss + self.distill_weight * kl
        return loss, logs

    def __call__(
        self,
        q: torch.Tensor,
        d: torch.Tensor,
        positive_index: Sequence[Sequence[int]],
        query_keys: Sequence[str],
        document_keys: Sequence[str],
        teacher_q: torch.Tensor | None = None,
        teacher_d: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        q, d = q.float(), d.float()
        pos = positive_mask(positive_index, d.size(0), q.device)
        dup = duplicate_doc_mask(positive_index, document_keys, q.device)
        qdup = duplicate_query_mask(query_keys, q.device)
        teacher_sims = None
        if teacher_q is not None and teacher_d is not None:
            teacher_sims = teacher_q.float() @ teacher_d.float().T
        loss, logs = self._single(q, d, pos, dup, qdup, teacher_sims)
        dims = [m for m in self.matryoshka_dims if m < q.size(1)]
        for dim in dims:
            extra, _ = self._single(F.normalize(q[:, :dim], dim=-1), F.normalize(d[:, :dim], dim=-1), pos, dup, qdup, None)
            logs[f"nce@{dim}"] = float(extra.detach())
            loss = loss + extra
        if dims:
            loss = loss / (1 + len(dims))
        logs["loss"] = float(loss.detach())
        return loss, logs
