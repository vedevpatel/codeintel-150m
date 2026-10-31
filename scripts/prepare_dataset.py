"""Build evaluation benchmarks and training data.

Subcommands:
  clone          clone the repositories in a manifest (configs/data/repos.yaml) and record their HEADs
  issues         mine an issue -> PR-changed-code localization benchmark from `eval` repos (GitHub API via gh)
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
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import yaml

from codeintel.data import validation
from codeintel.data.licenses import LicenseChecker
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
from codeintel.indexing.repository import detect_language


def _normalize_url(url: str) -> str:
    return url.lower().rstrip("/").removesuffix(".git")


def github_slug(url: str) -> str:
    return "/".join(_normalize_url(url).split("/")[-2:])


def load_manifest(path: str) -> dict:
    with open(path) as f:
        manifest = yaml.safe_load(f)
    missing = [r.get("name", "?") for r in manifest["repos"] if not str(r.get("license", "")).strip()]
    if missing:
        raise SystemExit(f"{path}: repos without a declared SPDX licence: {missing}")
    # Held-out evaluation repositories must never also be training sources.
    roles: dict[str, set[str]] = {}
    for r in manifest["repos"]:
        roles.setdefault(_normalize_url(r["url"]), set()).add(r["role"])
    overlap = sorted(url for url, rs in roles.items() if len(rs) > 1)
    names = Counter(r["name"] for r in manifest["repos"])
    duplicates = sorted(n for n, c in names.items() if c > 1)
    if overlap or duplicates:
        raise SystemExit(f"{path}: repos listed under more than one role {overlap}; duplicate names {duplicates}")
    return manifest


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
            # Per-repo override (train repos mined for older issues need more history); null = full clone.
            since = repo.get("shallow_since", manifest.get("shallow_since"))
            if since and not args.full:
                cmd[3:3] = [f"--shallow-since={since}"]
            print("+", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)
        head = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        problem = LicenseChecker(dest, repo["license"]).problem(head)
        if problem and repo.get("role") == "train":
            print(f"WARNING {repo['name']}: {problem} -- `pairs` will skip it", flush=True)
        lock[repo["name"]] = {"url": repo["url"], "head": head, "license": repo["license"], "license_problem": problem}
    (out / "repos.lock.json").write_text(json.dumps(lock, indent=2))
    print(f"recorded HEADs and licences in {out / 'repos.lock.json'}")
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


MINING_VERSION = 1  # bump when the miner's labelling rules change, to invalidate cached mined cases


def _case_language(case: LocalizationCase) -> str | None:
    """Language of the gold code. Mixed-language changes are ambiguous and excluded from per-language strata."""
    langs = {"python" if detect_language(f) == "python" else "typescript" for f in case.gold_files}
    return langs.pop() if len(langs) == 1 else None


def _mined_cases(repo: dict, args) -> tuple[list[LocalizationCase], dict]:
    """All usable issue cases of one repo, from the cache or freshly mined (the slow step)."""
    from codeintel.data.github import load_or_fetch
    from codeintel.data.mining import mine_issue_cases

    path = Path(args.mined_dir) / f"{repo['name']}.jsonl"
    params = {"since": args.since, "max_source_files": args.max_source_files, "version": MINING_VERSION}
    if path.exists() and not args.remine:
        header, *rows = path.read_text().splitlines()
        header = json.loads(header)
        if all(header.get(k) == v for k, v in params.items()):
            return [LocalizationCase(**json.loads(r)) for r in rows], header.get("stats", {})
    links = load_or_fetch(github_slug(repo["url"]), args.since, args.github_cache, refresh=args.refresh)
    cases, stats = mine_issue_cases(
        Path(args.repos_dir) / repo["name"], repo["name"], links, since=args.since, max_source_files=args.max_source_files
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(json.dumps({**params, "repo": repo["name"], "stats": vars(stats)}) + "\n")
        for c in cases:
            f.write(json.dumps(asdict(c)) + "\n")
    return cases, vars(stats)


def _stratum_report(cases: list[LocalizationCase], floor: int, min_repos: int, max_share: float, min_per_repo: int) -> bool:
    """Print per-language stratum sizes and check the descriptive-stratum conditions. Returns True if all pass."""
    ok = True
    for language in sorted({c.language for c in cases}):
        in_lang = [c for c in cases if c.language == language]
        counts = Counter(c.tags["mentions"] for c in in_lang)
        desc = Counter(c.repo for c in in_lang if c.tags["mentions"] == "descriptive")
        total = sum(desc.values())
        contributing = [r for r, n in desc.items() if n >= min_per_repo]
        top_repo, top_n = desc.most_common(1)[0] if desc else ("-", 0)
        share = top_n / total if total else 0.0
        checks = {
            f"descriptive >= {floor}": total >= floor,
            f"repos with >= {min_per_repo} descriptive >= {min_repos}": len(contributing) >= min_repos,
            f"largest repo share <= {max_share:.0%}": share <= max_share,
        }
        ok &= all(checks.values())
        print(f"[{language}] {len(in_lang)} cases {dict(counts)} across {len({c.repo for c in in_lang})} repos")
        print(f"    descriptive={total}, repos with any={len(desc)}, with >={min_per_repo}={len(contributing)}, "
              f"largest={top_repo} {top_n} ({share:.0%})")
        for name, passed in checks.items():
            print(f"    {'PASS' if passed else 'FAIL'}  {name}")
    return ok


def cmd_issues(args) -> int:
    """Issue -> PR localization: issue text as query, the PR's changed functions (pre-merge) as gold.

    Mining is cached per repository (--mined-dir); sampling caps cases per (repository, stratum) so that no
    repository dominates a stratum, and the descriptive-stratum floor conditions are reported per language.
    """
    manifest = load_manifest(args.manifest)
    languages = set(args.languages.split(",")) if args.languages else None
    wanted = set(args.repos.split(",")) if args.repos else None
    caps = {"descriptive": args.cap_descriptive, "same-file": args.cap_same_file, "lexical": args.cap_lexical}
    cases: list[LocalizationCase] = []
    for repo in repos_with_role(manifest, args.role):
        if (languages and repo["language"] not in languages) or (wanted and repo["name"] not in wanted):
            continue
        mined, stats = _mined_cases(repo, args)
        rng = random.Random(f"{args.seed}:{repo['name']}")
        by_stratum: dict[tuple[str, str], list[LocalizationCase]] = defaultdict(list)
        for c in sorted(mined, key=lambda c: c.id):
            language = _case_language(c)
            if language is None:
                continue
            c.language = language
            c.tags = {"language": language, "mentions": c.tags["mentions"]}
            by_stratum[(language, c.tags["mentions"])].append(c)
        kept = []
        for (_, mention), group in sorted(by_stratum.items()):
            kept += rng.sample(group, min(caps[mention], len(group)))
        cases.extend(sorted(kept, key=lambda c: c.id))
        available = Counter(c.tags["mentions"] for g in by_stratum.values() for c in g)
        print(f"{repo['name']}: {len(mined)} mined {dict(available)} -> kept {len(kept)} "
              f"{dict(Counter(c.tags['mentions'] for c in kept))}", flush=True)
    write_jsonl(args.out, cases)
    ok = _stratum_report(cases, args.floor, args.min_repos, args.max_share, args.min_per_repo)
    report = validation.validate_localization(args.out, args.repos_dir)
    print(report.summary())
    return 0 if report.ok and (ok or not args.require_floor) else 1


def cmd_resplit(args) -> int:
    """Re-split mined training pairs by repository, holding named repositories out as the dev set.

    Pairs are never re-mined: the input is the train/dev files written by `pairs`. A held-out repository's
    pairs go entirely to dev, so dev measures generalisation to unseen repositories.
    """
    from codeintel.data.schemas import TrainingPair

    src = Path(args.data)
    pairs = [p for name in ("train", "dev") if (src / f"{name}.jsonl").exists() for p in read_records(src / f"{name}.jsonl", TrainingPair)]
    dev = set(args.dev_repos.split(","))
    known = {p.repo for p in pairs}
    if dev - known:
        raise SystemExit(f"unknown repositories {sorted(dev - known)}; present: {sorted(known)}")
    splits = split_by_group(pairs, lambda p: p.repo, ratios={"train": 1.0, "dev": 0.0}, pinned={r: "dev" for r in dev})
    assert not group_leakage(splits, lambda p: p.repo)
    for name, recs in splits.items():
        write_jsonl(src / f"{name}.jsonl", recs)
        print(f"{name}: {len(recs)} pairs {dict(Counter(r.source for r in recs))} repos={sorted({r.repo for r in recs})}")
    return 0


def cmd_docstrings(args) -> int:
    """Docstring -> function retrieval over held-out repos: queries are doc summaries, corpus = all functions, docs removed."""
    manifest = load_manifest(args.manifest)
    rng = random.Random(args.seed)
    languages = set(args.languages.split(",")) if args.languages else None
    corpus, queries, qrels = [], [], []
    for repo in repos_with_role(manifest, args.role):
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


def cmd_code2code(args) -> int:
    """Code -> code retrieval over held-out repos: a test function -> the implementation it exercises (`tests`), or
    a call site -> the called definition (`calls`). The corpus is every non-test definition of the repository."""
    from codeintel.data.public import collapse_duplicates

    manifest = load_manifest(args.manifest)
    rng = random.Random(args.seed)
    languages = set(args.languages.split(",")) if args.languages else None
    corpus, queries, qrels = [], [], []
    for repo in repos_with_role(manifest, args.role):
        if languages and repo["language"] not in languages:
            continue
        mined = mine_snapshot(snapshot_files(Path(args.repos_dir) / repo["name"]), repo["name"])
        pairs = dedupe_pairs(mined.tests if args.kind == "tests" else mined.calls)
        docs: dict[str, str] = {}
        for p in mined.docstring:
            docs.setdefault(p.positive, p.id)
        for c in mined.undocumented:
            docs.setdefault(c.text, f"{repo['name']}:{c.chunk_id}")
        for p in pairs:
            docs.setdefault(p.positive, f"{repo['name']}:target:{len(docs)}")
        corpus += [CorpusDoc(id=i, text=t, language=repo["language"], repo=repo["name"]) for t, i in docs.items()]
        counts = Counter(p.query for p in pairs)
        pairs = [p for p in pairs if counts[p.query] == 1]
        for p in rng.sample(pairs, min(args.per_repo, len(pairs))):
            qid = f"q:{p.id}"
            queries.append(QueryRecord(id=qid, text=p.query, language=p.language, repo=p.repo))
            qrels.append(Qrel(query_id=qid, doc_id=docs[p.positive]))
        print(f"{repo['name']}: {len(pairs)} {args.kind} pairs, {len(docs)} definitions", flush=True)
    corpus, qrels, merged = collapse_duplicates(corpus, qrels)
    out = Path(args.out)
    write_jsonl(out / "corpus.jsonl", corpus)
    write_jsonl(out / "queries.jsonl", queries)
    write_jsonl(out / "qrels.jsonl", qrels)
    report = validation.validate_retrieval_dir(out)
    print(f"merged {merged} duplicate documents;", report.summary())
    return 0 if report.ok else 1


def cmd_import_coir(args) -> int:
    from codeintel.data.public import import_codesearchnet, import_coir_native, import_cosqa

    out = Path(args.out)
    for lang in args.csn_languages.split(","):
        print(f"codesearchnet-{lang}:", import_codesearchnet(lang, out / f"csn-{lang}"), flush=True)
    print("cosqa:", import_cosqa(out / "cosqa"), flush=True)
    # CoIR tasks exactly as published (query/corpus direction and full corpora), under <out>/coir/.
    coir = [("CoIR-Retrieval/stackoverflow-qa", "stackoverflow-qa", None), ("CoIR-Retrieval/apps", "apps", None)]
    for lang in ("python", "javascript"):
        coir += [("CoIR-Retrieval/CodeSearchNet", f"csn-native-{lang}", lang), ("CoIR-Retrieval/CodeSearchNet-ccr", f"csn-ccr-{lang}", lang)]
    for repo, name, lang in coir:
        print(f"coir {name}:", import_coir_native(repo, out / "coir" / name, lang), flush=True)
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
    skipped: dict[str, str] = {}
    for repo in repos_with_role(manifest, "train"):
        repo_dir = Path(args.repos_dir) / repo["name"]
        # Only permissively licensed code is trained on, checked at every revision code is taken from.
        licence = LicenseChecker(repo_dir, repo["license"])
        if (problem := licence.problem(args.snapshot_rev)) is not None:
            skipped[repo["name"]] = problem
            print(f"{repo['name']}: SKIPPED ({problem})", flush=True)
            continue
        repo_pairs = []
        mined = mine_snapshot(snapshot_files(repo_dir, rev=args.snapshot_rev), repo["name"])
        repo_pairs += mined.docstring + mined.tests + mined.calls
        dropped_commits = 0
        for ex in mine_commits(repo_dir, until=args.until):
            if licence.problem(ex.commit.parent) is not None:
                dropped_commits += 1
                continue
            repo_pairs += ex.to_pairs(repo["name"])
        if args.issues:
            from codeintel.data.github import load_or_fetch
            from codeintel.data.mining import issue_training_pairs, mine_issue_cases

            links = load_or_fetch(github_slug(repo["url"]), args.issues_since, args.github_cache)
            labels: dict = {}
            cases, _ = mine_issue_cases(repo_dir, repo["name"], links, since=args.issues_since, labels_out=labels)
            dropped_issues = 0
            for case in cases:
                if licence.problem(case.commit) is not None:
                    dropped_issues += 1
                    continue
                repo_pairs += issue_training_pairs(case, labels[case.id])
            if dropped_issues:
                print(f"{repo['name']}: {dropped_issues} issue cases dropped (licence at PR base)", flush=True)
        for p in repo_pairs:
            p.license = repo["license"]
        pairs += repo_pairs
        note = f", {dropped_commits} commits dropped (licence at parent)" if dropped_commits else ""
        print(f"{repo['name']} [{repo['license']}]: {len(repo_pairs)} pairs{note}", flush=True)
    if skipped:
        print(f"skipped {len(skipped)} train repos for licence reasons: {skipped}")
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
    p.add_argument("--role", choices=["eval", "train", "dev"], default="eval", help="train: a dev set from training repos; dev: the development repositories")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--out", default="data/eval/temporal-localization/cases.jsonl")
    p.set_defaults(func=cmd_localization)

    p = sub.add_parser("issues")
    p.add_argument("--manifest", default="configs/data/repos.yaml")
    p.add_argument("--repos-dir", default="data/repos")
    p.add_argument("--github-cache", default="data/github")
    p.add_argument("--refresh", action="store_true", help="re-query the GitHub API instead of using the cache")
    p.add_argument("--languages", help="comma-separated filter, e.g. python")
    p.add_argument("--since", default="2025-03-01", help="issues (and their PRs) created on/after this date")
    p.add_argument("--repos", help="comma-separated manifest names (default: every repo of the role)")
    p.add_argument("--role", choices=["eval", "train", "dev"], default="eval",
                   help="train: build a development set from held-out training repositories")
    p.add_argument("--mined-dir", default="data/eval/issue-mined", help="per-repo cache of all mined cases")
    p.add_argument("--remine", action="store_true", help="ignore the mined-case cache")
    p.add_argument("--cap-descriptive", type=int, default=75, help="per repo and language; 25%% of the 300 floor")
    p.add_argument("--cap-same-file", type=int, default=40)
    p.add_argument("--cap-lexical", type=int, default=60)
    p.add_argument("--floor", type=int, default=300, help="descriptive cases required per language")
    p.add_argument("--min-repos", type=int, default=8, help="repos that must contribute descriptive cases")
    p.add_argument("--min-per-repo", type=int, default=10, help="descriptive cases for a repo to count as contributing")
    p.add_argument("--max-share", type=float, default=0.25, help="largest repo's share of the descriptive stratum")
    p.add_argument("--require-floor", action="store_true", help="exit non-zero if the floor conditions fail")
    p.add_argument("--max-source-files", type=int, default=10, help="skip PRs changing more source files")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--out", default="data/eval/issue-localization/cases.jsonl")
    p.set_defaults(func=cmd_issues)

    p = sub.add_parser("resplit", help="hold named repositories out of training as the dev set")
    p.add_argument("--data", default="data/train")
    p.add_argument("--dev-repos", required=True, help="comma-separated repository names")
    p.set_defaults(func=cmd_resplit)

    p = sub.add_parser("docstrings")
    p.add_argument("--manifest", default="configs/data/repos.yaml")
    p.add_argument("--repos-dir", default="data/repos")
    p.add_argument("--languages", help="comma-separated filter, e.g. typescript")
    p.add_argument("--per-repo", type=int, default=150)
    p.add_argument("--role", choices=["eval", "dev"], default="eval", help="repositories to build the task from")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_docstrings)

    p = sub.add_parser("code2code", help="test -> implementation or call site -> definition retrieval")
    p.add_argument("--manifest", default="configs/data/repos.yaml")
    p.add_argument("--repos-dir", default="data/repos")
    p.add_argument("--kind", choices=["tests", "calls"], required=True)
    p.add_argument("--languages", help="comma-separated filter, e.g. python")
    p.add_argument("--role", choices=["eval", "dev"], default="eval")
    p.add_argument("--per-repo", type=int, default=60)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_code2code)

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
    p.add_argument("--issues", action="store_true", help="also mine issue -> changed-function pairs via the GitHub API")
    p.add_argument("--issues-since", default="2023-01-01")
    p.add_argument("--github-cache", default="data/github")
    p.add_argument("--out", default="data/train")
    p.set_defaults(func=cmd_pairs)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
