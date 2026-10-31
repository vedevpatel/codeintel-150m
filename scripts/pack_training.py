"""Pack a training mixture into one compact file for the GPU machine.

  uv run python scripts/pack_training.py --config configs/training/codeintel.yaml --out artifacts/train/groups.json.gz \
      --train data/train/v4/train.jsonl ... --cap tests=30000 calls=9000

Inputs are mined pairs (with hard negatives) and public-dataset pairs; only sources with a share
in the config are kept. Low-share sources are capped: a run draws steps x batch group slots (3,000 x 128 = 384k), so a
source with a 3% share needs about 12k distinct groups, not 100k.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import yaml

from codeintel.training.datasets import group_pairs, load_training_pairs
from codeintel.training.pack import pack_groups

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", default=[str(ROOT / "data/train/train.hn.jsonl")])
    ap.add_argument("--config", default=str(ROOT / "configs/training/codeintel.yaml"))
    ap.add_argument("--out", default=str(ROOT / "artifacts/train/groups.json.gz"))
    ap.add_argument("--cap", nargs="*", default=["tests=4000", "calls=6000", "docstring=40000", "commits=12000"],
                    help="source=max_groups")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    shares = yaml.safe_load(Path(args.config).read_text())["source_shares"]
    groups = [g for g in group_pairs(load_training_pairs(args.train)) if g.source in shares]
    print("groups by source:", dict(Counter(g.source for g in groups)))
    caps = {k: int(v) for k, v in (c.split("=") for c in args.cap)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    stats = pack_groups(groups, args.out, per_source_cap=caps, seed=args.seed)
    print(stats)
    with_neg = sum(1 for g in groups if g.negatives)
    print(f"{with_neg}/{len(groups)} groups carry mined negatives; {stats['bytes'] / 1e6:.1f} MB packed")


if __name__ == "__main__":
    main()
