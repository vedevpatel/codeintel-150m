"""Precomputed teacher embeddings for distillation.

The distillation term only needs the teacher's similarity matrix over a batch, so the teacher's embeddings of every
training query and document can be computed once (one pass over the packed data) instead of at every step. The
teacher may differ from the student in architecture, tokenizer, prompts and dimension: only the dot products of
its own (normalised) embeddings are used.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class TeacherCache:
    query_index: dict[str, int]
    query_emb: torch.Tensor  # (n_queries, dim) fp16, L2-normalised
    doc_index: dict[str, int]
    doc_emb: torch.Tensor  # (n_docs, dim) fp16
    model: str = ""

    def _rows(self, index: dict[str, int], emb: torch.Tensor, texts: Sequence[str], device) -> torch.Tensor:
        try:
            rows = [index[t] for t in texts]
        except KeyError as exc:
            raise KeyError(f"text missing from the teacher cache (built for different data?): {str(exc)[:80]}") from None
        return emb[torch.as_tensor(rows)].to(device=device, dtype=torch.float32)

    def queries(self, texts: Sequence[str], device) -> torch.Tensor:
        return self._rows(self.query_index, self.query_emb, texts, device)

    def documents(self, texts: Sequence[str], device) -> torch.Tensor:
        return self._rows(self.doc_index, self.doc_emb, texts, device)

    def save(self, path: str | Path) -> None:
        torch.save({"queries": list(self.query_index), "query_emb": self.query_emb, "docs": list(self.doc_index),
                    "doc_emb": self.doc_emb, "model": self.model}, path)

    @classmethod
    def load(cls, path: str | Path) -> TeacherCache:
        data = torch.load(path, map_location="cpu", weights_only=True)
        return cls({t: i for i, t in enumerate(data["queries"])}, data["query_emb"],
                   {t: i for i, t in enumerate(data["docs"])}, data["doc_emb"], data.get("model", ""))


def build_teacher_cache(model, queries: Sequence[str], documents: Sequence[str]) -> TeacherCache:
    """Encode unique texts with `model` (its own prompts) and keep fp16 copies."""
    q = list(dict.fromkeys(queries))
    d = list(dict.fromkeys(documents))
    qe = torch.from_numpy(np.asarray(model.encode_queries(q), dtype=np.float32)).half()
    de = torch.from_numpy(np.asarray(model.encode_documents(d), dtype=np.float32)).half()
    return TeacherCache({t: i for i, t in enumerate(q)}, qe, {t: i for i, t in enumerate(d)}, de,
                        getattr(model, "identity", getattr(model, "name", "")))
