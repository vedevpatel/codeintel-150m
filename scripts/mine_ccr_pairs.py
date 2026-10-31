"""Code-context pairs (CoIR CodeSearchNet-ccr style): the first part of a function -> the rest of it.

  uv run python scripts/mine_ccr_pairs.py --train data/train/v2/train.jsonl --out data/train/v4/ccr.jsonl

Functions come from the merged, decontaminated training pairs (docstring and call-site positives; permissive training
repositories only). Each function of at least --min-lines lines is split on a line boundary at a seeded fraction in
[0.3, 0.6]; the query keeps the `# path::Symbol` header and the opening lines, the positive is the remaining lines.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

from codeintel.data.schemas import TrainingPair, read_records, write_jsonl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train/v2/train.jsonl")
    ap.add_argument("--out", default="data/train/v4/ccr.jsonl")
    ap.add_argument("--min-lines", type=int, default=8)
    ap.add_argument("--per-repo", type=int, default=600)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    seen, by_repo = set(), {}
    for p in read_records(args.train, TrainingPair):
        if p.source in ("docstring", "calls") and p.positive not in seen:
            seen.add(p.positive)
            by_repo.setdefault(p.repo, []).append(p)
    pairs = []
    for repo, items in sorted(by_repo.items()):
        rng = random.Random(f"{args.seed}:{repo}")
        rng.shuffle(items)
        kept = 0
        for p in items:
            header, _, body = p.positive.partition("\n")
            lines = body.splitlines()
            if not header.startswith(("# ", "// ")) or len(lines) < args.min_lines:
                continue
            cut = max(2, min(len(lines) - 2, round(len(lines) * rng.uniform(0.3, 0.6))))
            query = header + "\n" + "\n".join(lines[:cut])
            rest = "\n".join(lines[cut:])
            if len(rest.strip()) < 40:
                continue
            pairs.append(TrainingPair(query=query, positive=rest, task="code2code", source="ccr", language=p.language,
                                      repo=repo, id=f"{p.id}:ccr", license=p.license, meta={"cut": cut}))
            kept += 1
            if kept >= args.per_repo:
                break
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    print({"pairs": write_jsonl(args.out, pairs), "repos": len({p.repo for p in pairs}),
           "languages": dict(Counter(p.language for p in pairs))})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
