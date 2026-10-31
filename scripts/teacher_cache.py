"""Compute a teacher-embedding cache for the texts of a packed training file.

  python scripts/teacher_cache.py --model configs/models/coderankembed.yaml --packed train.json.gz --out teacher.pt

Queries are encoded as queries and documents (positives and negatives) as documents, each with the teacher's own
prompts. The trainer uses it when `teacher_cache` is set (distillation without a live teacher forward pass).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

from codeintel.config import load_model_config
from codeintel.models.base import load_model
from codeintel.training.pack import unpack_groups
from codeintel.training.teacher import build_teacher_cache


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--packed", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-length", type=int, default=512, help="the training document length")
    ap.add_argument("--dtype", default="auto")
    args = ap.parse_args()
    t0 = time.time()
    groups = unpack_groups(args.packed)
    model = load_model(replace(load_model_config(args.model), max_length=args.max_length, dtype=args.dtype))
    queries = [g.query for g in groups]
    docs = [t for g in groups for t in (*g.positives, *g.negatives)]
    cache = build_teacher_cache(model, queries, docs)
    cache.save(args.out)
    print(f"teacher cache: {len(cache.query_index)} queries, {len(cache.doc_index)} documents, dim {cache.doc_emb.shape[1]}, "
          f"{(time.time() - t0) / 60:.1f} min -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
