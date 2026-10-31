"""Turn a GPU job's log into local evaluation checkpoints.

  uv run python scripts/collect_job_log.py artifacts/logs/test.log --out artifacts/eval/run --cases <cases.jsonl>

Writes <out>/<variant>/checkpoints/issue-localization/<repo>.json for every complete ranking block in the log (a job
that was cut off still yields every repository it finished) and reports which repositories are missing. Then
`codeintel evaluate <config> --output-dir <out>/<variant>` resumes from those checkpoints and aggregates locally
without embedding anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from codeintel.data.schemas import LocalizationCase, read_records
from codeintel.evaluation.transport import parse_stream


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--out", default="artifacts/eval/issue-v1")
    ap.add_argument("--task", default="issue-localization")
    ap.add_argument("--cases", default="data/eval/issue-localization/cases.jsonl")
    args = ap.parse_args()

    found_all, problems = parse_stream(Path(args.log).read_text(errors="replace"))
    # Blocks from a variant run are labelled "variant:repo" and go to <out>/<variant>/checkpoints/...
    by_variant: dict[str, dict] = {}
    for label, checkpoint in found_all.items():
        variant, _, repo = label.rpartition(":")
        by_variant.setdefault(variant, {})[repo] = checkpoint
    expected = sorted({c.repo for c in read_records(args.cases, LocalizationCase)})
    missing: dict[str, list[str]] = {}
    for variant, found in sorted(by_variant.items()):
        out = Path(args.out) / variant / "checkpoints" / args.task if variant else Path(args.out) / "checkpoints" / args.task
        out.mkdir(parents=True, exist_ok=True)
        for repo, checkpoint in found.items():
            (out / f"{repo}.json").write_text(json.dumps(checkpoint))
        print(f"recovered {len(found)}/{len(expected)} repositories for variant {variant or 'default'} into {out}")
        if gaps := [r for r in expected if r not in found]:
            missing[variant or "default"] = gaps
    for p in problems:
        print(f"  problem: {p}")
    for variant, gaps in missing.items():
        print(f"  missing for {variant}: {', '.join(gaps)}")
    if missing:
        print("  re-run with --only-repos for the missing repositories; the aggregate would otherwise re-score them locally")
    print(f"next: uv run codeintel evaluate configs/evaluation/issue-localization.yaml --output-dir {args.out}")
    return 0 if not problems and not missing else 1


if __name__ == "__main__":
    sys.exit(main())
