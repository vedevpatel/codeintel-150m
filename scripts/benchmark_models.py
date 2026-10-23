"""Benchmark candidate checkpoints: retrieval quality (via an evaluation config) and encode throughput.

  uv run python scripts/benchmark_models.py --config configs/evaluation/baseline.yaml --out artifacts/benchmarks/baseline
  uv run python scripts/benchmark_models.py --skip-eval --throughput --devices cpu mps
  uv run python scripts/benchmark_models.py --config configs/evaluation/int8.yaml \
      --extra-models onnx:artifacts/onnx/gte-modernbert:int8 onnx:artifacts/onnx/gte-modernbert:float32

`--extra-models` accepts `onnx:<export dir>:<int8|float32>` to score exported graphs with the same tasks.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

from codeintel.config import ModelConfig, load_evaluation_config, load_model_config
from codeintel.evaluation.runner import run_evaluation
from codeintel.indexing.chunker import Chunker
from codeintel.indexing.repository import iter_source_files
from codeintel.models.base import load_model

ROOT = Path(__file__).resolve().parent.parent
CANDIDATES = [ROOT / "configs/models/coderankembed.yaml", ROOT / "configs/models/gte-modernbert.yaml"]


def load_any(spec: str):
    if spec.startswith("onnx:"):
        _, directory, variant = spec.split(":", 2)
        name = f"{Path(directory).name}-onnx-{variant}"
        cfg = ModelConfig(name=name, adapter="onnx", repo_id="local", local_dir=directory, dtype=variant, max_length=1024)
        return load_model(cfg)
    return load_model(load_model_config(spec))


def sample_texts(n: int) -> list[str]:
    """Deterministic code chunks from this repository's own sources (always available offline)."""
    chunker = Chunker()
    texts = [c.text for f in iter_source_files(ROOT / "src") for c in chunker.chunk(f)]
    while len(texts) < n:
        texts = texts + texts
    return texts[:n]


def throughput(model_paths: list[str], devices: list[str], dtypes: list[str], n_docs: int) -> list[dict]:
    texts = sample_texts(n_docs)
    queries = ["where is the retry backoff implemented", "parse a yaml configuration file", "reciprocal rank fusion"]
    rows = []
    for path in model_paths:
        for device in devices:
            for dtype in dtypes:
                cfg = replace(load_model_config(path), device=device, dtype=dtype)
                try:
                    model = load_model(cfg)
                except (RuntimeError, ValueError) as exc:
                    rows.append({"model": cfg.name, "device": device, "dtype": dtype, "error": str(exc)})
                    continue
                model.encode_documents(texts[:16])  # warm-up
                t = time.perf_counter()
                model.encode_documents(texts)
                docs_per_s = len(texts) / (time.perf_counter() - t)
                lat = []
                for _ in range(10):
                    for q in queries:
                        t = time.perf_counter()
                        model.encode_queries([q])
                        lat.append((time.perf_counter() - t) * 1000)
                rows.append(
                    {
                        "model": cfg.name,
                        "device": device,
                        "dtype": dtype,
                        "docs_per_s": round(docs_per_s, 1),
                        "query_ms_p50": round(statistics.median(lat), 1),
                        "query_ms_p95": round(sorted(lat)[int(0.95 * len(lat)) - 1], 1),
                    }
                )
                print(rows[-1], flush=True)
                del model
                gc.collect()
    return rows


def render_throughput(rows: list[dict]) -> str:
    lines = ["## Throughput", "", "| model | device | dtype | docs/s | query p50 ms | query p95 ms |", "|---|---|---|---|---|---|"]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['model']} | {r['device']} | {r['dtype']} | error: {r['error']} | | |")
        else:
            lines.append(f"| {r['model']} | {r['device']} | {r['dtype']} | {r['docs_per_s']} | {r['query_ms_p50']} | {r['query_ms_p95']} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(ROOT / "configs/evaluation/baseline.yaml"))
    parser.add_argument("--out", default=str(ROOT / "artifacts/benchmarks/latest"))
    parser.add_argument("--extra-models", nargs="*", default=[])
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--throughput", action="store_true")
    parser.add_argument("--devices", nargs="*", default=["cpu"])
    parser.add_argument("--dtypes", nargs="*", default=["float32"])
    parser.add_argument("--throughput-docs", type=int, default=256)
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sections = []
    if not args.skip_eval:
        cfg = load_evaluation_config(args.config)
        cfg.models = list(cfg.models) + args.extra_models
        run_evaluation(cfg, model_loader=load_any, output_dir=out)
        sections.append((out / "report.md").read_text())
    if args.throughput:
        rows = throughput([str(p) for p in CANDIDATES], args.devices, args.dtypes, args.throughput_docs)
        (out / "throughput.json").write_text(json.dumps(rows, indent=2))
        sections.append(render_throughput(rows))
    (out / "summary.md").write_text("\n\n".join(sections))
    print(f"wrote {out / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
