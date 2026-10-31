"""Run the localization evaluation in parallel worker processes on one machine.

A single evaluation process alternates between GPU embedding and single-threaded Python (git, chunking, BM25,
ranking), so the GPU idles during CPU phases and the CPU idles during GPU phases. Sharding the repositories
across worker processes overlaps the two and uses every vCPU; repositories are independent, so the workers
share nothing but the GPU.

Workers print each finished repository's checkpoint (`transport.pack`). `BlockRelay` forwards worker output
to the parent's stdout, emitting each checkpoint block atomically so blocks from different workers never
interleave.

  python -m codeintel.evaluation.shard --cases C --repos-dir R --workdir W --models a.yaml,b.yaml --dtypes '{...}'
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import IO

from codeintel.evaluation.transport import BEGIN, END

# Zero-shot variants, named so a job and the local aggregation agree on what each label means.
VARIANT_PRESETS: dict[str, dict] = {
    "baseline": {},
    "clean": {"query_cleaning": True},
    "header": {"chunk_context": "rich"},
    "both": {"chunk_context": "rich", "query_cleaning": True},
}


def assign_shards(costs: Mapping[str, float], n: int) -> list[list[str]]:
    """Greedy longest-processing-time assignment of repositories to `n` workers; empty shards are dropped."""
    shards: list[list[str]] = [[] for _ in range(max(1, n))]
    loads = [0.0] * len(shards)
    for repo, cost in sorted(costs.items(), key=lambda kv: (-kv[1], kv[0])):
        i = loads.index(min(loads))
        shards[i].append(repo)
        loads[i] += cost
    return [sorted(s) for s in shards if s]


class BlockRelay:
    """Thread-safe forwarding of worker output; checkpoint blocks are written whole, other lines are tagged."""

    def __init__(self, out: IO[str]):
        self.out = out
        self.lock = threading.Lock()
        self.truncated: list[str] = []

    def _write(self, text: str) -> None:
        with self.lock:
            self.out.write(text)
            self.out.flush()

    def pump(self, name: str, stream: IO[str]) -> None:
        block: list[str] | None = None
        for line in stream:
            if line.startswith(BEGIN):
                block = [line]
            elif block is not None:
                block.append(line)
                if line.startswith(END):
                    self._write("".join(block))
                    block = None
            else:
                self._write(f"[{name}] {line}")
        if block is not None:  # the worker died inside a block; the partial block is useless without its CRC
            self.truncated.append(name)
            self._write(f"[{name}] output ended inside a checkpoint block; it was dropped\n")


def worker_main(argv: list[str] | None = None) -> int:
    from codeintel.config import EvaluationConfig, TaskConfig, load_model_config
    from codeintel.evaluation.runner import run_evaluation
    from codeintel.evaluation.transport import pack
    from codeintel.models.base import load_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--repos-dir", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--models", required=True, help="comma-separated model YAML paths")
    ap.add_argument("--dtypes", default="{}", help='JSON {model yaml path: dtype}; missing entries use "auto"')
    ap.add_argument("--task", default="issue-localization")
    ap.add_argument("--label", default="worker")
    ap.add_argument("--modes", default="dense,bm25,hybrid", help='retrieval modes; "hybrid@2.0" sets the dense RRF weight')
    ap.add_argument("--variants", default="", help=f"comma-separated subset of {sorted(VARIANT_PRESETS)}; default: baseline only")
    args = ap.parse_args(argv)

    from codeintel.data.schemas import LocalizationCase, read_records

    n_repos = len({c.repo for c in read_records(args.cases, LocalizationCase)})
    dtypes = json.loads(args.dtypes)
    workdir = Path(args.workdir)
    names = [v for v in args.variants.split(",") if v]
    unknown = [v for v in names if v not in VARIANT_PRESETS]
    if unknown:
        raise SystemExit(f"unknown variants {unknown}; choose from {sorted(VARIANT_PRESETS)}")
    models: dict[str, object] = {}

    def loader(path: str):
        if path not in models:  # variants share loaded models across runs
            models[path] = load_model(replace(load_model_config(path), dtype=dtypes.get(path, "auto")))
        return models[path]

    for name in names or [""]:
        cfg = EvaluationConfig(
            models=args.models.split(","),
            tasks=[TaskConfig(name=args.task, type="localization", path=args.cases, repos_dir=args.repos_dir,
                              contamination_note="temporal-issues")],
            output_dir=str(workdir / "out" / (name or "baseline")), ks=[1, 5, 10, 20, 50],
            retrieval_modes=args.modes.split(","), primary_metric="ndcg@10", bootstrap_samples=50, seed=13,
            cache_dir=str(workdir / "cache"), **VARIANT_PRESETS.get(name, {}),
        )
        done: list[str] = []

        def on_repo_done(repo: str, checkpoint: dict, name=name, done=done) -> None:
            done.append(repo)
            label = f"{name}:{repo}" if name else repo  # the collector splits variant from repository on ":"
            sys.stdout.write(pack(label, checkpoint))  # one write: never split by other output of this process
            sys.stdout.write(f"[{name or 'baseline'}] repository {len(done)}/{n_repos} done: {repo}\n")
            sys.stdout.flush()

        run_evaluation(cfg, model_loader=loader, on_repo_done=on_repo_done)
    return 0


if __name__ == "__main__":
    raise SystemExit(worker_main())
