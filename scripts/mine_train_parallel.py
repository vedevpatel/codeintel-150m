"""Mine training pairs from every `train` repository in parallel, one cached file per repository.

  uv run python scripts/mine_train_parallel.py --workers 6            # resumable: finished repositories are skipped
  uv run python scripts/mine_train_parallel.py --merge --out data/train/v2

Per repository (licence verified at every revision code is taken from, as in `prepare_dataset.py pairs`):
docstring / test / call-site pairs at HEAD, commit pairs before --until, issue -> edited-function pairs from the
cached GitHub links, then hard negatives (BM25 hits + same-file chunks from one HEAD index per repository, never
another version of any edited symbol) for issue groups and up to --max-commit-queries commit groups. `--merge` concatenates the per-repository files, removes
duplicates, decontaminates against every evaluation AND development text, and blocks eval/dev repositories.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_dataset import github_slug, load_manifest  # noqa: E402

# Training repositories held out as in-training development data; never trained on.
HELD_OUT = {"gradio", "typeorm"}


def mine_repo(repo: dict, args_dict: dict) -> str:
    from codeintel.data.github import load_or_fetch
    from codeintel.data.licenses import LicenseChecker
    from codeintel.data.mining import (
        issue_training_pairs,
        mine_commits,
        mine_issue_cases,
        mine_snapshot,
        snapshot_files,
    )
    from codeintel.data.schemas import write_jsonl
    from codeintel.evaluation.runner import SnapshotLoader
    from codeintel.indexing.chunker import Chunker
    from codeintel.retrieval.query import clean_issue_query
    from codeintel.training.negatives import mine_group_negatives

    t0 = time.time()
    name = repo["name"]
    repos_dir = Path(args_dict["repos_dir"])
    repo_dir = repos_dir / name
    out = Path(args_dict["per_repo"]) / f"{name}.jsonl"
    licence = LicenseChecker(repo_dir, repo["license"])
    if (problem := licence.problem("HEAD")) is not None:
        out.with_suffix(".skipped").write_text(problem + "\n")
        return f"{name}: SKIPPED ({problem})"
    pairs = []
    mined = mine_snapshot(snapshot_files(repo_dir), name)
    pairs += mined.docstring + mined.tests + mined.calls
    dropped = 0
    for ex in mine_commits(repo_dir, until=args_dict["until"], sample=args_dict["max_commits"], seed=13):
        if licence.problem(ex.commit.parent) is not None:
            dropped += 1
            continue
        pairs += ex.to_pairs(name)
    links = load_or_fetch(github_slug(repo["url"]), args_dict["issues_since"], args_dict["github_cache"])
    labels: dict = {}
    cases, _ = mine_issue_cases(repo_dir, name, links, since=args_dict["issues_since"], labels_out=labels)
    n_cases = 0
    for case in cases:
        if licence.problem(case.commit) is not None:
            continue
        pairs += issue_training_pairs(case, labels[case.id])
        n_cases += 1
    for p in pairs:
        p.license = repo["license"]

    # Hard negatives for issue groups (all) and a capped, random subset of commit groups.
    rng = random.Random(f"13:{name}")
    by_query = defaultdict(list)
    for p in pairs:
        by_query[(p.source, p.query)].append(p)
    keys = sorted(by_query)
    commit_keys = [k for k in keys if k[0] == "commits"]
    rng.shuffle(commit_keys)
    chosen = [k for k in keys if k[0] == "issues"][: args_dict["max_issue_queries"]] + commit_keys[: args_dict["max_commit_queries"]]
    # One BM25 index per repository (HEAD) instead of one per base commit: building an index per snapshot dominated
    # the run time on large repositories. Other versions of every positive (same path::Symbol) are excluded.
    from codeintel.training.negatives import positive_labels

    head = subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    loader = SnapshotLoader(repos_dir, Chunker())
    index = loader.load(name, head)
    bm25 = loader.bm25(index)
    all_labels = positive_labels(p.positive for p in pairs if p.source in ("issues", "commits"))
    with_negs = 0
    for key in chosen:
        group = by_query[key]
        query = clean_issue_query(key[1]) if key[0] == "issues" else key[1]
        positives = [p.positive for p in group]
        negs = mine_group_negatives(index.chunks, bm25, query, positives, set(positives), rng,
                                    banned_labels=positive_labels(positives) | all_labels)
        for p in group:
            p.negatives = negs
        with_negs += bool(negs)
    tmp = out.with_suffix(".tmp")
    write_jsonl(tmp, pairs)
    tmp.replace(out)
    by_source = Counter(p.source for p in pairs)
    return (f"{name}: {len(pairs)} pairs {dict(by_source)}; {n_cases} issue cases from {len(links)} links; "
            f"{with_negs} groups with negatives; {dropped} commits dropped (licence); {(time.time() - t0) / 60:.1f} min")


def merge(args) -> int:
    from codeintel.data import validation
    from codeintel.data.mining import dedupe_pairs
    from codeintel.data.schemas import (
        CorpusDoc,
        LocalizationCase,
        QueryRecord,
        TrainingPair,
        read_records,
        write_jsonl,
    )
    from codeintel.evaluation.contamination import decontaminate

    manifest = load_manifest(args.manifest)
    pairs = []
    for f in sorted(Path(args.per_repo).glob("*.jsonl")):
        pairs += read_records(f, TrainingPair)
    if args.include:
        for f in args.include:  # previously mined pairs, restricted to current train repositories
            pairs += [p for p in read_records(f, TrainingPair) if p.repo not in HELD_OUT]
    print(f"{len(pairs)} pairs before dedupe", flush=True)
    pairs = dedupe_pairs(pairs)
    texts: list[str] = []
    for path in args.decontaminate:
        path = Path(path)
        if path.is_dir():
            texts += [d.text for d in read_records(path / "corpus.jsonl", CorpusDoc)]
            texts += [q.text for q in read_records(path / "queries.jsonl", QueryRecord)]
        else:
            texts += [c.query for c in read_records(path, LocalizationCase)]
    blocked = {r["name"] for r in manifest["repos"] if r["role"] in ("eval", "dev")} | HELD_OUT
    pairs, stats = decontaminate(pairs, texts, blocked_repos=blocked)
    print("decontamination:", stats, flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "train.jsonl", pairs)
    print(f"train: {len(pairs)} pairs {dict(Counter(p.source for p in pairs))}; repos {len({p.repo for p in pairs})}")
    report = validation.validate_training(out / "train.jsonl")
    print(report.summary())
    return 0 if report.ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="configs/data/repos.yaml")
    ap.add_argument("--repos-dir", default="data/repos")
    ap.add_argument("--github-cache", default="data/github")
    ap.add_argument("--per-repo", default="data/train/per-repo")
    ap.add_argument("--issues-since", default="2023-01-01")
    ap.add_argument("--until", default="2025-01-01", help="commit pairs only from commits before this date")
    ap.add_argument("--max-commits", type=int, default=400, help="random sample of commit examples per repository")
    ap.add_argument("--max-issue-queries", type=int, default=600)
    ap.add_argument("--max-commit-queries", type=int, default=300)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", help="comma-separated repository names")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--include", nargs="*", default=[])
    ap.add_argument("--decontaminate", nargs="*", default=[
        "data/eval/issue-localization/cases.jsonl", "data/eval/docstrings-python", "data/eval/docstrings-typescript",
        "data/eval/cosqa", "data/eval/csn-python", "data/eval/csn-javascript", "data/dev/combined/cases.jsonl",
        "data/dev/retention/cosqa", "data/dev/retention/docstrings-python", "data/dev/retention/docstrings-typescript",
        "data/eval/code2code-tests-python", "data/eval/code2code-tests-typescript", "data/eval/code2code-calls-python",
        "data/eval/code2code-calls-typescript"])
    ap.add_argument("--out", default="data/train/v2")
    args = ap.parse_args()
    if args.merge:
        return merge(args)

    manifest = load_manifest(args.manifest)
    Path(args.per_repo).mkdir(parents=True, exist_ok=True)
    only = set(args.only.split(",")) if args.only else None
    todo = []
    for r in manifest["repos"]:
        if r["role"] != "train" or r["name"] in HELD_OUT or (only and r["name"] not in only):
            continue
        done = Path(args.per_repo) / f"{r['name']}.jsonl"
        cache = Path(args.github_cache) / f"{github_slug(r['url']).replace('/', '__')}.jsonl"
        if done.exists() or done.with_suffix(".skipped").exists():
            continue
        if not (Path(args.repos_dir) / r["name"] / ".git").exists():
            print(f"{r['name']}: not cloned yet, skipped this round", flush=True)
            continue
        if not cache.exists() or json.loads(cache.read_text().splitlines()[0]).get("since") != args.issues_since:
            print(f"{r['name']}: GitHub links not fetched yet, skipped this round", flush=True)
            continue
        todo.append(r)
    print(f"{len(todo)} repositories to mine with {args.workers} workers", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        futures = {pool.submit(mine_repo, r, vars(args)): r["name"] for r in todo}
        for fut in as_completed(futures):
            try:
                print(fut.result(), flush=True)
            except Exception:  # noqa: BLE001
                print(f"{futures[fut]}: FAILED\n{traceback.format_exc()[-1500:]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
