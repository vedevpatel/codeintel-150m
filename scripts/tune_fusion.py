"""Choose each model's hybrid (RRF) dense weight on the dev repositories.

  uv run python scripts/tune_fusion.py artifacts/jobs/dev.log --cases data/dev/localization/cases.jsonl \
      --out artifacts/eval/dev-fusion --write configs/evaluation/fusion_weights.json

Procedure: for each model, the weight with the highest mean
symbol-level nDCG@10 over all dev cases; ties go to the weight closest to 1. Eval repositories are never read.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from codeintel.config import EvaluationConfig, TaskConfig, load_model_config
from codeintel.evaluation.runner import run_evaluation
from codeintel.models.base import load_model

GRID = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
MODELS = {"coderankembed": "configs/models/coderankembed.yaml", "gte-modernbert": "configs/models/gte-modernbert.yaml"}


def fmt(w: float) -> str:
    return f"{w:g}"


def mode(w: float) -> str:
    """Retrieval-mode string for a dense weight; 1 is plain "hybrid"."""
    return "hybrid" if w == 1.0 else f"hybrid@{fmt(w)}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--cases", default="data/dev/localization/cases.jsonl")
    ap.add_argument("--out", default="artifacts/eval/dev-fusion")
    ap.add_argument("--repos-dir", default="data/repos")
    ap.add_argument("--write", help="write the chosen weights to this JSON file")
    ap.add_argument("--model", action="append", help="model YAML to tune (repeatable; default: the two zero-shot bases)")
    args = ap.parse_args()
    models = {load_model_config(p).name: p for p in args.model} if args.model else MODELS

    out = Path(args.out)
    subprocess.run([sys.executable, "scripts/collect_job_log.py", args.log, "--out", str(out), "--cases", args.cases], check=False)
    modes = ["dense", "bm25"] + [mode(w) for w in GRID]
    cfg = EvaluationConfig(
        models=list(models.values()),
        tasks=[TaskConfig(name="issue-localization", type="localization", path=args.cases, repos_dir=args.repos_dir)],
        output_dir=str(out / "clean"), ks=[1, 5, 10, 20, 50], bootstrap_samples=50, cache_dir=str(out / "cache"),
        retrieval_modes=modes, query_cleaning=True,
    )
    res = run_evaluation(cfg, model_loader=lambda p: load_model(load_model_config(p)))
    stats = res["tasks"]["issue-localization"]["stats"]
    if stats["repos_resumed_from_checkpoint"] != stats["repos"]:
        print("WARNING: not every repository came from checkpoints; results below include local re-scoring")

    scores: dict[str, list[float]] = {}
    for line in open(out / "clean" / "per_query.jsonl"):
        row = json.loads(line)
        if row["level"] == "symbol":
            scores.setdefault(row["system"], []).append(row["ndcg@10"])
    n = len(scores["bm25"])
    print(f"\nmean symbol-level nDCG@10 over {n} dev cases (bm25 {sum(scores['bm25']) / n:.4f})")
    chosen = {}
    print(f"{'model':16s} {'dense':>7s} " + " ".join(f"w={fmt(w):>5s}" for w in GRID))
    for model in models:
        means = {w: sum(scores[f"{model}/{mode(w)}"]) / n for w in GRID}
        best = max(GRID, key=lambda w: (round(means[w], 12), -abs(w - 1.0)))
        chosen[model] = best
        dense = sum(scores[f"{model}/dense"]) / n
        print(f"{model:16s} {dense:7.4f} " + " ".join(f"{means[w]:7.4f}" for w in GRID) + f"   -> w={fmt(best)}")
    if args.write:
        Path(args.write).write_text(json.dumps({"dense_weight": chosen, "grid": GRID, "dev_cases": n}, indent=2) + "\n")
        print(f"wrote {args.write}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
