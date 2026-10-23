"""Build evaluation benchmarks and training data.

Subcommands:
  clone          clone the repositories in a manifest (configs/data/repos.yaml) and record their HEADs
  localization   mine a commit-message -> changed-code localization benchmark from `eval` repos
  docstrings     build a docstring -> function retrieval task from `eval` repos (covers TypeScript)
  import-coir    import CodeSearchNet (NL->code) and CoSQA from CoIR at pinned revisions
  pairs          mine training pairs from `train` repos, split by repo, decontaminate against eval tasks

Examples:
  uv run python scripts/prepare_dataset.py clone --manifest configs/data/repos.yaml --out data/repos
  uv run python scripts/prepare_dataset.py localization --manifest configs/data/repos.yaml --repos-dir data/repos \
      --since 2025-03-01 --per-repo 40 --out data/eval/temporal-localization/cases.jsonl
  uv run python scripts/prepare_dataset.py import-coir --out data/eval
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

import yaml

from codeintel.data import validation
from codeintel.data.mining import candidate_commits, dedupe_pairs, mine_commits, mine_snapshot, snapshot_files
from codeintel.data.schemas import (
    CorpusDoc,
    LocalizationCase,
    Qrel,
    QueryRecord,
    TrainingPair,
    read_records,
    write_jsonl,
)
from codeintel.data.splits import group_leakage, split_by_group
from codeintel.evaluation.contamination import decontaminate


def load_manifest(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def repos_with_role(manifest: dict, role: str) -> list[dict]:
    return [r for r in manifest["repos"] if r.get("role") == role]


def cmd_clone(args) -> int:
    manifest = load_manifest(args.manifest)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lock = {}
    for repo in manifest["repos"]:
        dest = out / repo["name"]
        if not dest.exists():
            cmd = ["git", "clone", "--quiet", "--single-branch", repo["url"], str(dest)]
            if manifest.get("shallow_since") and not args.full:
                cmd[3:3] = [f"--shallow-since={manifest['shallow_since']}"]
            print("+", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)
        head = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        lock[repo["name"]] = {"url": repo["url"], "head": head}
    (out / "repos.lock.json").write_text(json.dumps(lock, indent=2))
    print(f"recorded HEADs in {out / 'repos.lock.json'}")
    return 0


def cmd_localization(args) -> int:
    manifest = load_manifest(args.manifest)
    cases: list[LocalizationCase] = []
    for repo in repos_with_role(manifest, args.role):
        repo_dir = Path(args.repos_dir) / repo["name"]
        n_candidates = len(candidate_commits(repo_dir, args.since, args.until, args.max_gold_files))
        picked = mine_commits(
            repo_dir, since=args.since, until=args.until, max_gold_files=args.max_gold_files,
            sample=args.per_repo, seed=args.seed,
        )
        picked = sorted(picked, key=lambda e: e.commit.date)
        cases.extend(e.to_case(repo["name"]) for e in picked)
        print(f"{repo['name']}: {n_candidates} candidate commits, kept {len(picked)}", flush=True)
    for c in cases:
        c.language = next(r["language"] for r in manifest["repos"] if r["name"] == c.repo)
    write_jsonl(args.out, cases)
    report = validation.validate_localization(args.out, args.repos_dir)
    print(report.summary())
    return 0 if report.ok else 1


def cmd_docstrings(args) -> int:
    """Docstring -> function retrieval over held-out repos: queries are doc summaries, corpus = all functions, docs removed."""
    manifest = load_manifest(args.manifest)
    rng = random.Random(args.seed)
    languages = set(args.languages.split(",")) if args.languages else None
    corpus, queries, qrels = [], [], []
    for repo in repos_with_role(manifest, "eval"):
        if languages and repo["language"] not in languages:
            continue
        repo_dir = Path(args.repos_dir) / repo["name"]
        mined = mine_snapshot(snapshot_files(repo_dir), repo["name"])
        pairs = dedupe_pairs(mined.docstring)
        # Keep one pair per query text so that duplicate summaries cannot make the task ambiguous.
        counts = Counter(p.query for p in pairs)
        pairs = [p for p in pairs if counts[p.query] == 1]
        for p in pairs:
            corpus.append(CorpusDoc(id=p.id, text=p.positive, language=p.language, repo=p.repo))
        # Undocumented functions as distractors: the corpus is then every function, none with a doc comment.
        for c in mined.undocumented:
            corpus.append(CorpusDoc(id=f"{repo['name']}:{c.chunk_id}", text=c.text, language=c.language, repo=repo["name"]))
        for p in rng.sample(pairs, min(args.per_repo, len(pairs))):
            qid = f"q:{p.id}"
            queries.append(QueryRecord(id=qid, text=p.query, language=p.language, repo=p.repo))
            qrels.append(Qrel(query_id=qid, doc_id=p.id))
        print(f"{repo['name']}: {len(pairs)} documented + {len(mined.undocumented)} undocumented functions", flush=True)
    out = Path(args.out)
    write_jsonl(out / "corpus.jsonl", corpus)
    write_jsonl(out / "queries.jsonl", queries)
    write_jsonl(out / "qrels.jsonl", qrels)
    report = validation.validate_retrieval_dir(out)
    print(report.summary())
    return 0 if report.ok else 1


def cmd_import_coir(args) -> int:
    from codeintel.data.public import import_codesearchnet, import_cosqa

    out = Path(args.out)
    for lang in args.csn_languages.split(","):
        print(f"codesearchnet-{lang}:", import_codesearchnet(lang, out / f"csn-{lang}"), flush=True)
    print("cosqa:", import_cosqa(out / "cosqa"), flush=True)
    return 0


def _eval_texts(paths: list[str]) -> list[str]:
    texts: list[str] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            texts += [d.text for d in read_records(path / "corpus.jsonl", CorpusDoc)]
            texts += [q.text for q in read_records(path / "queries.jsonl", QueryRecord)]
        else:
            texts += [c.query for c in read_records(path, LocalizationCase)]
    return texts


def cmd_pairs(args) -> int:
    manifest = load_manifest(args.manifest)
    pairs: list[TrainingPair] = []
    for repo in repos_with_role(manifest, "train"):
        repo_dir = Path(args.repos_dir) / repo["name"]
        mined = mine_snapshot(snapshot_files(repo_dir, rev=args.snapshot_rev), repo["name"])
        pairs += mined.docstring + mined.tests + mined.calls
        for ex in mine_commits(repo_dir, until=args.until):
            pairs += ex.to_pairs(repo["name"])
        print(f"{repo['name']}: {len(pairs)} pairs so far", flush=True)
    pairs = dedupe_pairs(pairs)
    eval_repos = {r["name"] for r in repos_with_role(manifest, "eval")}
    pairs, stats = decontaminate(pairs, _eval_texts(args.eval_tasks), blocked_repos=eval_repos)
    print("decontamination:", stats)
    splits = split_by_group(pairs, lambda p: p.repo, ratios={"train": 0.95, "dev": 0.05})
    assert not group_leakage(splits, lambda p: p.repo)
    out = Path(args.out)
    for name, recs in splits.items():
        by_source = Counter(r.source for r in recs)
        write_jsonl(out / f"{name}.jsonl", recs)
        print(f"{name}: {len(recs)} pairs {dict(by_source)}")
    report = validation.validate_training(out / "train.jsonl")
    print(report.summary())
    return 0 if report.ok else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("clone")
    p.add_argument("--manifest", default="configs/data/repos.yaml")
    p.add_argument("--out", default="data/repos")
    p.add_argument("--full", action="store_true", help="full history instead of --shallow-since")
    p.set_defaults(func=cmd_clone)

    p = sub.add_parser("localization")
    p.add_argument("--manifest", default="configs/data/repos.yaml")
    p.add_argument("--repos-dir", default="data/repos")
    p.add_argument("--since", default="2025-03-01")
    p.add_argument("--until")
    p.add_argument("--per-repo", type=int, default=40)
    p.add_argument("--max-gold-files", type=int, default=5)
    p.add_argument("--role", choices=["eval", "train"], default="eval", help="train: a dev set from training repos")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--out", default="data/eval/temporal-localization/cases.jsonl")
    p.set_defaults(func=cmd_localization)

    p = sub.add_parser("docstrings")
    p.add_argument("--manifest", default="configs/data/repos.yaml")
    p.add_argument("--repos-dir", default="data/repos")
    p.add_argument("--languages", help="comma-separated filter, e.g. typescript")
    p.add_argument("--per-repo", type=int, default=150)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_docstrings)

    p = sub.add_parser("import-coir")
    p.add_argument("--out", default="data/eval")
    p.add_argument("--csn-languages", default="python,javascript")
    p.set_defaults(func=cmd_import_coir)

    p = sub.add_parser("pairs")
    p.add_argument("--manifest", default="configs/data/repos.yaml")
    p.add_argument("--repos-dir", default="data/repos")
    p.add_argument("--snapshot-rev", default="HEAD")
    p.add_argument("--until", default="2025-01-01", help="only mine commits before this date")
    p.add_argument("--eval-tasks", nargs="*", default=[], help="retrieval dirs / localization files to decontaminate against")
    p.add_argument("--out", default="data/train")
    p.set_defaults(func=cmd_pairs)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
