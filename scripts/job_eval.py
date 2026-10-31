"""Issue-localization evaluation as a self-contained GPU job (template for scripts/build_job.py).

The job prints everything it produces: each repository's rankings are streamed as a checksummed block
(`codeintel.evaluation.transport`) as soon as that repository is scored, and recovered locally with
scripts/collect_job_log.py, even from a job that was cut off.

Order of work, cheapest and most likely to fail first: hardware report -> unpack -> dependencies -> clone repositories
-> verify every case's base commit -> per-model precision and throughput preflight (abort if the projected embedding
time exceeds --abort-if-over-minutes) -> evaluation.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PAYLOAD = ""  # base64(tar.gz) of src/, configs/, cases.jsonl, chunk_counts.json; filled in by build_job.py


def log(msg: str) -> None:
    print(f"[job {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def hardware_report() -> None:
    import platform

    try:
        mem_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (ValueError, OSError):
        mem_gb = float("nan")
    log(f"python {platform.python_version()} on {platform.platform()}; {os.cpu_count()} vCPU, {mem_gb:.1f} GB RAM")
    try:
        import torch

        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            log(f"torch {torch.__version__}; GPU {p.name}, {p.total_memory / 2**30:.1f} GB, compute capability {p.major}.{p.minor}")
        else:
            log(f"torch {torch.__version__}; no CUDA device (using {'mps' if torch.backends.mps.is_available() else 'cpu'})")
    except ImportError:
        log("torch is not installed yet")
    log(f"disk free: {shutil.disk_usage('/').free / 2**30:.0f} GB")


def unpack_payload(workdir: Path) -> Path:
    root = workdir / "bundle"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD)), mode="r:gz") as tar:
        tar.extractall(root, filter="data")
    log(f"unpacked bundle: {sum(1 for _ in root.rglob('*') if _.is_file())} files")
    return root


def install_dependencies() -> None:
    # JOB_TRANSFORMERS pins another transformers version (reference models whose remote code predates v5).
    packages = [os.environ.get("JOB_TRANSFORMERS", "transformers>=4.48"), "einops", "tree-sitter>=0.23", "tree-sitter-typescript>=0.23", "pyyaml",
                "safetensors", "huggingface-hub", "numpy"]
    t = time.time()
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *packages], check=True)
    log(f"dependencies ready in {time.time() - t:.0f}s")


def clone_repos(manifest: dict, wanted: set[str], repos_dir: Path) -> None:
    repos_dir.mkdir(parents=True, exist_ok=True)
    by_name = {r["name"]: r for r in manifest["repos"]}

    def clone(name: str) -> str:
        dest = repos_dir / name
        if dest.exists():
            return f"{name}: present"
        repo = by_name[name]
        since = repo.get("shallow_since", manifest.get("shallow_since"))
        cmd = ["git", "clone", "--quiet", "--single-branch", repo["url"], str(dest)]
        if since:
            cmd[3:3] = [f"--shallow-since={since}"]
        for attempt in range(3):
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode == 0:
                return f"{name}: cloned"
            shutil.rmtree(dest, ignore_errors=True)
            time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"clone of {name} failed: {proc.stderr.strip()[-300:]}")

    t = time.time()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for line in pool.map(clone, sorted(wanted)):
            log(line)
    log(f"{len(wanted)} repositories ready in {time.time() - t:.0f}s")


def verify_base_commits(cases, repos_dir: Path) -> None:
    missing = []
    for case in cases:
        rc = subprocess.run(["git", "-C", str(repos_dir / case.repo), "cat-file", "-e", f"{case.commit}^{{commit}}"],
                            capture_output=True).returncode
        if rc != 0:
            missing.append(case.id)
    if missing:
        raise SystemExit(f"{len(missing)} base commits are missing from the clones (first: {missing[:5]}); aborting before any GPU work")
    log(f"all {len(cases)} base commits present")


def sample_real_chunks(repos_dir: Path, repos: list[str], n: int = 512) -> list[str]:
    """An evenly spaced sample of chunk texts from the selected repositories (the real length distribution)."""
    from codeintel.indexing.chunker import Chunker
    from codeintel.indexing.repository import iter_source_files

    chunker = Chunker()
    texts: list[str] = []
    for repo in sorted(repos)[:3]:
        texts += [c.text for f in iter_source_files(repos_dir / repo) for c in chunker.chunk(f)]
    step = max(1, len(texts) // n)
    return texts[::step][:n]


def preflight(model_paths: list[str], expected_chunks: int, abort_minutes: float | None, texts_pool: list[str]):
    """Pick each model's dtype by cosine agreement with fp32, and measure throughput. Returns {path: dtype}."""
    from dataclasses import replace

    import numpy as np
    import torch

    from codeintel.config import load_model_config
    from codeintel.models.base import cuda_has_native_bf16, load_model

    chosen = {}
    on_cuda = torch.cuda.is_available()
    if not on_cuda:
        log("preflight: no CUDA device, keeping dtype 'auto'")
        return {p: "auto" for p in model_paths}
    texts = (texts_pool * 2)[:512]
    log(f"preflight sample: {len(texts)} chunks, mean {sum(len(t) for t in texts) / len(texts):.0f} chars")
    total_minutes = 0.0
    for path in model_paths:
        cfg = load_model_config(path)
        cfg = replace(cfg, device="cuda")
        ref = load_model(replace(cfg, dtype="float32")).encode_documents(texts[:64])
        candidates = (["bfloat16"] if cuda_has_native_bf16() else []) + ["float16"]
        pick, rate = "float32", None
        for dtype in candidates:
            model = load_model(replace(cfg, dtype=dtype))
            cos = float(np.min(np.sum(model.encode_documents(texts[:64]) * ref, axis=1)))
            model.encode_documents(texts[:32])  # warm-up
            torch.cuda.synchronize()
            t = time.perf_counter()
            model.encode_documents(texts)
            torch.cuda.synchronize()
            r = len(texts) / (time.perf_counter() - t)
            log(f"preflight {cfg.name}: {dtype:9s} min cosine vs fp32 {cos:.6f}, {r:.0f} chunks/s")
            if cos >= 0.9999 and pick == "float32":
                pick, rate = dtype, r
            del model
            torch.cuda.empty_cache()
        if rate is None:
            model = load_model(replace(cfg, dtype="float32"))
            model.encode_documents(texts[:32])
            torch.cuda.synchronize()
            t = time.perf_counter()
            model.encode_documents(texts)
            torch.cuda.synchronize()
            rate = len(texts) / (time.perf_counter() - t)
            log(f"preflight {cfg.name}: no half-precision dtype matched fp32, using float32 at {rate:.0f} chunks/s")
            del model
        minutes = expected_chunks / rate / 60
        total_minutes += minutes
        log(f"preflight {cfg.name}: dtype {pick}; projected embedding time for {expected_chunks:,} chunks: {minutes:.0f} min")
        chosen[path] = pick
    log(f"projected GPU embedding time, all models: {total_minutes:.0f} min (CPU phases come on top)")
    if abort_minutes is not None and total_minutes > abort_minutes:
        raise SystemExit(f"projected {total_minutes:.0f} min exceeds --abort-if-over-minutes {abort_minutes:.0f}; aborting")
    return chosen


def run_sharded(root, selected_cases, repos_dir, workdir, model_paths, chosen, shards, counts, cases, t_start,
                variants: str = "", modes: str = "dense,bm25,hybrid") -> int:
    """Evaluate repositories in parallel worker processes; the parent relays their checkpoint blocks."""
    import threading

    from codeintel.evaluation.shard import BlockRelay, assign_shards

    per_repo: dict[str, int] = {}
    for c in cases:
        per_repo[c.repo] = per_repo.get(c.repo, 0) + 1
    # Cost model: GPU embedding scales with unique chunks, CPU phases with cases.
    costs = {r: counts.get(r, {}).get("unique_chunks", 10_000) + 150 * n for r, n in per_repo.items()}
    groups = assign_shards(costs, shards)
    log(f"{len(groups)} workers; repositories per worker: {[len(g) for g in groups]}")
    relay = BlockRelay(sys.stdout)
    stop_heartbeat = threading.Event()

    def heartbeat() -> None:
        """A line a minute, so a stalled or lost session is visible in the log within minutes."""
        while not stop_heartbeat.wait(60):
            alive = sum(1 for p in procs if p.poll() is None)
            log(f"heartbeat: {alive}/{len(procs)} workers alive, {(time.time() - t_start) / 60:.0f} min elapsed")
    env = dict(os.environ, PYTHONPATH=str(root / "src"), PYTHONUNBUFFERED="1")
    procs, threads = [], []
    for i, repos_i in enumerate(groups):
        cases_i = workdir / f"cases.shard{i}.jsonl"
        cases_i.write_text("".join(json.dumps(c.__dict__) + "\n" for c in cases if c.repo in set(repos_i)))
        cmd = [sys.executable, "-m", "codeintel.evaluation.shard", "--cases", str(cases_i), "--repos-dir", str(repos_dir),
               "--workdir", str(workdir / f"shard{i}"), "--models", ",".join(model_paths), "--dtypes", json.dumps(chosen),
               "--label", f"w{i}", "--variants", variants, "--modes", modes]
        proc = subprocess.Popen(cmd, env=env, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        t = threading.Thread(target=relay.pump, args=(f"w{i}", proc.stdout), daemon=True)
        t.start()
        procs.append(proc)
        threads.append(t)
    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        codes = [p.wait() for p in procs]
    except BaseException:
        for p in procs:
            p.kill()
        raise
    finally:
        stop_heartbeat.set()
    for t in threads:
        t.join(timeout=30)
    log(f"workers exited with {codes}; finished in {(time.time() - t_start) / 60:.1f} min")
    return 0 if all(c == 0 for c in codes) and not relay.truncated else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="/content/job")
    ap.add_argument("--repos-dir", help="use existing clones here instead of cloning (local dry runs)")
    ap.add_argument("--only-repos", help="comma-separated repository names")
    ap.add_argument("--limit-per-repo", type=int, help="first N cases per repository (smoke tests)")
    ap.add_argument("--models", default="coderankembed,gte-modernbert")
    ap.add_argument("--abort-if-over-minutes", type=float, help="abort after preflight if projected GPU time is higher")
    ap.add_argument("--no-install", action="store_true")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--profile", action="store_true", help="cProfile the evaluation and print the hottest functions")
    ap.add_argument("--modes", default="dense,bm25,hybrid", help='retrieval modes, e.g. dense,bm25,hybrid@0.5,hybrid@2')
    ap.add_argument("--variants", default="", help="zero-shot variants to run, e.g. baseline,clean,header,both")
    ap.add_argument("--shards", type=int, help="parallel worker processes (default: min(vCPUs, 6); 1 = in-process)")
    args = ap.parse_args()

    t_start = time.time()
    workdir = Path(args.workdir).resolve()
    given_repos_dir = Path(args.repos_dir).resolve() if args.repos_dir else None  # before any chdir
    workdir.mkdir(parents=True, exist_ok=True)
    hardware_report()
    root = unpack_payload(workdir)
    if not args.no_install:
        install_dependencies()
    sys.path.insert(0, str(root / "src"))
    os.chdir(root)

    import yaml

    from codeintel.config import EvaluationConfig, TaskConfig, load_model_config
    from codeintel.data.schemas import LocalizationCase, read_records
    from codeintel.evaluation.runner import render_report, run_evaluation
    from codeintel.evaluation.transport import pack
    from codeintel.models.base import load_model

    manifest = yaml.safe_load((root / "configs/data/repos.yaml").read_text())
    cases = read_records(root / "cases.jsonl", LocalizationCase)
    if args.only_repos:
        keep = set(args.only_repos.split(","))
        cases = [c for c in cases if c.repo in keep]
    if args.limit_per_repo:
        seen: dict[str, int] = {}
        limited = []
        for c in sorted(cases, key=lambda c: (c.repo, c.created_at, c.id)):
            seen[c.repo] = seen.get(c.repo, 0) + 1
            if seen[c.repo] <= args.limit_per_repo:
                limited.append(c)
        cases = limited
    repos = {c.repo for c in cases}
    log(f"{len(cases)} cases over {len(repos)} repositories")
    selected_cases = workdir / "cases.selected.jsonl"
    selected_cases.write_text("".join(json.dumps(c.__dict__) + "\n" for c in cases))

    repos_dir = given_repos_dir or workdir / "repos"
    if not args.repos_dir:
        clone_repos(manifest, repos, repos_dir)
    verify_base_commits(cases, repos_dir)

    counts = json.loads((root / "chunk_counts.json").read_text())
    expected = sum(counts[r]["unique_chunks"] for r in repos if r in counts)
    # A name resolves to configs/models/<name>.yaml; a path ending in .yaml is used as given (fine-tuned exports).
    model_paths = [m if m.endswith(".yaml") else str(root / f"configs/models/{m}.yaml") for m in args.models.split(",")]
    pool = sample_real_chunks(repos_dir, sorted(repos))
    chosen = preflight(model_paths, expected, args.abort_if_over_minutes, pool)
    if args.preflight_only:
        log("preflight only: stopping before evaluation")
        return 0

    shards = args.shards or max(1, min(os.cpu_count() or 1, 6))
    if (shards > 1 or args.variants) and not args.profile:
        return run_sharded(root, selected_cases, repos_dir, workdir, model_paths, chosen, shards, counts, cases, t_start,
                           args.variants, args.modes)

    cfg = EvaluationConfig(
        models=model_paths,
        tasks=[TaskConfig(name="issue-localization", type="localization", path=str(selected_cases),
                          repos_dir=str(repos_dir), contamination_note="temporal-issues")],
        output_dir=str(workdir / "out"), ks=[1, 5, 10, 20, 50], retrieval_modes=["dense", "bm25", "hybrid"],
        primary_metric="ndcg@10", bootstrap_samples=1000, seed=13, cache_dir=str(workdir / "cache"),
    )
    from dataclasses import replace

    def loader(path: str):
        mc = replace(load_model_config(path), dtype=chosen[path])
        return load_model(mc)

    done = []

    def on_repo_done(repo: str, checkpoint: dict) -> None:
        done.append(repo)
        sys.stdout.write(pack(repo, checkpoint))
        log(f"repository {len(done)}/{len(repos)} done: {repo} (elapsed {(time.time() - t_start) / 60:.1f} min)")
        sys.stdout.flush()

    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    profiler = None
    if args.profile:
        import cProfile

        profiler = cProfile.Profile()
        profiler.enable()
    results = run_evaluation(cfg, model_loader=loader, on_repo_done=on_repo_done)
    if profiler is not None:
        import io
        import pstats

        profiler.disable()
        for order in ("tottime", "cumulative"):
            buf = io.StringIO()
            pstats.Stats(profiler, stream=buf).sort_stats(order).print_stats(28)
            print(f"@@PROFILE {order}")
            print("\n".join(line[:170] for line in buf.getvalue().splitlines() if line.strip()))
            print("@@END")
    print("@@REPORT")
    print(render_report(results))
    print("@@END")
    for system, t in results["tasks"]["issue-localization"]["timings"].items():
        log(f"timings {system}: {t}")
    log(f"finished in {(time.time() - t_start) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
