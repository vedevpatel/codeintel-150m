"""GPU job step: retrieval evaluation of a list of models on retrieval tasks (template for scripts/build_job.py,
which embeds src/, configs/ and, with --extra-dir, task directories). Arguments (JOB_ARGS):

  --models a.yaml,b.yaml   model YAMLs (a bare name resolves to configs/models/<name>.yaml)
  --tasks name=dir,...     task directories inside the bundle, e.g. cosqa=data/dev/retention/cosqa
  --modes dense,bm25,hybrid

JOB_DATA_TAR (optional) names an uploaded .tar.gz of further task directories (or its .partNNN pieces).

Prints one line `@@RETRIEVAL {json}` per task, with nDCG@10 and Recall@10 for each system.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

PAYLOAD = ""  # base64(tar.gz); filled in by build_job.py


def log(msg: str) -> None:
    print(f"[retrieval {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--modes", default="dense,bm25,hybrid")
    ap.add_argument("--workdir", default="/content/job/retrieval")
    args = ap.parse_args()
    root = Path(args.workdir) / "bundle"
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD)), mode="r:gz") as tar:
        tar.extractall(root, filter="data")
    if os.environ.get("JOB_DATA_TAR"):  # task data too large for the script: uploaded as a .tar.gz, possibly in parts
        data_tar = Path(os.environ["JOB_DATA_TAR"])
        if not data_tar.exists():
            parts = sorted(data_tar.parent.glob(data_tar.name + ".part*"))
            with open(data_tar, "wb") as joined:
                for part in parts:
                    joined.write(part.read_bytes())
        with tarfile.open(data_tar, mode="r:gz") as tar:
            tar.extractall(root, filter="data")
        log(f"extracted task data from {data_tar}")
    # JOB_TRANSFORMERS pins another transformers version (reference models whose remote code predates v5).
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", os.environ.get("JOB_TRANSFORMERS", "transformers>=4.48"), "einops", "tree-sitter>=0.23",
                    "tree-sitter-typescript>=0.23", "pyyaml", "safetensors", "huggingface-hub", "numpy"], check=True)
    sys.path.insert(0, str(root / "src"))
    os.chdir(root)

    from dataclasses import replace

    from codeintel.config import EvaluationConfig, TaskConfig, load_model_config
    from codeintel.evaluation.runner import run_evaluation
    from codeintel.models.base import cuda_has_native_bf16, load_model  # noqa: F401

    models = [m if m.endswith(".yaml") else str(root / f"configs/models/{m}.yaml") for m in args.models.split(",")]
    tasks = []
    for item in args.tasks.split(","):
        name, _, path = item.partition("=")
        tasks.append(TaskConfig(name=name, type="retrieval", path=str(root / path)))
    # One task at a time, each result printed as soon as it exists: a session cut off by its cap keeps every task it
    # finished (the embedding cache makes tasks that share a corpus cheap).
    t = time.time()
    for task in tasks:
        cfg = EvaluationConfig(models=models, tasks=[task], output_dir=str(Path(args.workdir) / "out" / task.name),
                               ks=[1, 5, 10, 20, 50], retrieval_modes=args.modes.split(","), primary_metric="ndcg@10",
                               bootstrap_samples=50, seed=13, cache_dir=str(Path(args.workdir) / "cache"))
        results = run_evaluation(cfg, model_loader=lambda p: load_model(replace(load_model_config(p), dtype="float16")))
        compact: dict = {}
        for name, res in results["tasks"].items():
            compact[name] = {system: {m: v for m, v in entry["metrics"].items() if m in ("ndcg@10", "recall@10", "mrr")}
                             for system, entry in res["levels"]["doc"]["systems"].items()}
        log(f"{task.name} done ({time.time() - t:.0f}s elapsed)")
        print("@@RETRIEVAL " + json.dumps(compact), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
