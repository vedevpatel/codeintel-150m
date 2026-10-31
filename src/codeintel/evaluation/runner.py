"""Run retrieval and localization benchmarks for one or more embedding models.

Systems evaluated per task:
  * `bm25`                 lexical baseline (model independent, run once)
  * `<model>/dense`        embedding retrieval with the model's own inference contract
  * `<model>/hybrid`       reciprocal-rank fusion of dense and BM25

Retrieval tasks are scored at document level. Localization tasks index the repository at the case's
commit, retrieve chunks, and score the aggregated file ranking and (when gold symbols exist) the
aggregated symbol ranking. Cases carrying `tags` (e.g. language, whether the query names gold identifiers)
are additionally scored per tag value, as levels named `symbol[mentions=descriptive]` etc.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import platform
import random
import subprocess
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codeintel.config import EvaluationConfig, TaskConfig, load_model_config
from codeintel.data.schemas import CorpusDoc, LocalizationCase, Qrel, QueryRecord, read_records
from codeintel.evaluation.contamination import KNOWN_EXPOSURE
from codeintel.evaluation.metrics import (
    bootstrap_ci,
    cluster_bootstrap_ci,
    clustered_paired_comparison,
    evaluate_rankings,
    macro_mean,
    paired_comparison,
)
from codeintel.indexing.chunker import Chunker, ChunkerConfig
from codeintel.indexing.repository import (
    DEFAULT_LANGUAGES,
    iter_source_files,
    list_source_blobs,
    read_source_blobs,
)
from codeintel.indexing.storage import CodeIndex, EmbeddingCache, chunk_files, encode_documents_cached
from codeintel.models.base import EmbeddingModel
from codeintel.retrieval.bm25 import BM25Index, code_tokens, term_arrays
from codeintel.retrieval.dense import DenseIndex
from codeintel.retrieval.hybrid import (
    Retriever,
    aggregate,
    file_key,
    reciprocal_rank_fusion,
    split_mode,
    symbol_key,
)
from codeintel.schemas import Chunk

logger = logging.getLogger(__name__)

LOCALIZATION_DEPTH = 1000  # chunks retrieved before aggregating to files / symbols
# Below this many repositories a repository-clustered bootstrap/permutation is degenerate (e.g. zero-width CIs
# with one repository); case-level statistics are used instead and labelled as such.
MIN_CLUSTERS = 5


# ------------------------------------------------------------------------------------------ results


@dataclass
class SystemResult:
    metrics: dict[str, float]
    per_query: dict[str, dict[str, float]]
    # query id -> repository; when present, CIs and comparisons resample repositories, and macro means are reported.
    clusters: dict[str, str] = field(default_factory=dict)


@dataclass
class TaskRun:
    """results[level][system] -> SystemResult, plus bookkeeping."""

    results: dict[str, dict[str, SystemResult]] = field(default_factory=lambda: defaultdict(dict))
    stats: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, dict[str, float]] = field(default_factory=lambda: defaultdict(dict))


# --------------------------------------------------------------------------------- retrieval tasks


@dataclass
class RetrievalData:
    corpus: list[CorpusDoc]
    queries: list[QueryRecord]
    qrels: dict[str, dict[str, float]]


def _load_qrels(directory: Path) -> list[Qrel]:
    if (directory / "qrels.jsonl").exists():
        return read_records(directory / "qrels.jsonl", Qrel)
    tsv = directory / "qrels.tsv"
    out = []
    with open(tsv, encoding="utf-8") as f:
        for i, line in enumerate(f):
            parts = line.rstrip("\n").split("\t")
            if i == 0 and not parts[-1].replace(".", "").isdigit():
                continue  # header
            out.append(Qrel(query_id=parts[0], doc_id=parts[1], relevance=float(parts[2])))
    return out


def load_retrieval_task(task: TaskConfig, seed: int = 13) -> RetrievalData:
    directory = Path(task.path)
    corpus = read_records(directory / "corpus.jsonl", CorpusDoc)
    queries = read_records(directory / "queries.jsonl", QueryRecord)
    qrels: dict[str, dict[str, float]] = defaultdict(dict)
    for q in _load_qrels(directory):
        if q.relevance > 0:
            qrels[q.query_id][q.doc_id] = q.relevance
    queries = [q for q in queries if q.id in qrels]
    rng = random.Random(seed)
    if task.max_queries and len(queries) > task.max_queries:
        queries = sorted(rng.sample(queries, task.max_queries), key=lambda q: q.id)
    keep_q = {q.id for q in queries}
    qrels = {qid: rels for qid, rels in qrels.items() if qid in keep_q}
    if task.max_corpus and len(corpus) > task.max_corpus:
        # Keep every judged document, then fill with a deterministic sample of distractors.
        needed = {d for rels in qrels.values() for d in rels}
        judged = [d for d in corpus if d.id in needed]
        others = [d for d in corpus if d.id not in needed]
        fill = max(0, task.max_corpus - len(judged))
        corpus = judged + rng.sample(others, min(fill, len(others)))
    return RetrievalData(corpus=corpus, queries=queries, qrels=qrels)


def _doc_text(doc: CorpusDoc) -> str:
    return f"{doc.title}\n{doc.text}" if doc.title else doc.text


def run_retrieval_task(
    data: RetrievalData,
    systems: Sequence[tuple[str, EmbeddingModel | None, str]],
    ks: Sequence[int],
    cache_dir: str | None,
    run: TaskRun,
) -> None:
    doc_ids = [d.id for d in data.corpus]
    texts = [_doc_text(d) for d in data.corpus]
    depth = max(max(ks), 100)
    bm25: BM25Index | None = None
    lexical: list[list[tuple[int, float]]] | None = None
    dense_cache: dict[int, list[list[tuple[int, float]]]] = {}

    def lexical_hits() -> list[list[tuple[int, float]]]:
        nonlocal bm25, lexical
        if lexical is None:
            t = time.perf_counter()
            bm25 = BM25Index(texts)
            lexical = [bm25.search(q.text, depth) for q in data.queries]
            run.timings["bm25"]["seconds"] = time.perf_counter() - t
        return lexical

    for name, model, mode in systems:
        if mode == "bm25":
            hits = lexical_hits()
        else:
            assert model is not None
            if id(model) not in dense_cache:
                cache = EmbeddingCache(cache_dir, model.fingerprint, model.dim) if cache_dir else None
                t = time.perf_counter()
                doc_emb = encode_documents_cached(model, texts, cache)
                t_docs = time.perf_counter() - t
                t = time.perf_counter()
                q_emb = model.encode_queries([q.text for q in data.queries])
                t_q = time.perf_counter() - t
                idx, scores = DenseIndex(doc_emb).search(q_emb, depth)
                dense_cache[id(model)] = [
                    [(int(i), float(s)) for i, s in zip(ri, rs)] for ri, rs in zip(idx, scores)
                ]
                run.timings[model.name].update(
                    {"encode_corpus_seconds": t_docs, "encode_queries_seconds": t_q, "num_docs": len(texts)}
                )
            hits = dense_cache[id(model)]
            if split_mode(mode)[0] == "hybrid":
                w = split_mode(mode)[1] or 1.0
                hits = [
                    reciprocal_rank_fusion([[i for i, _ in d], [i for i, _ in b]], weights=[w, 1.0])[:depth]
                    for d, b in zip(hits, lexical_hits())
                ]
        rankings = {q.id: [doc_ids[i] for i, _ in h] for q, h in zip(data.queries, hits)}
        means, per_query = evaluate_rankings(rankings, data.qrels, ks)
        clusters = {q.id: q.repo for q in data.queries} if all(q.repo for q in data.queries) else {}
        run.results["doc"][name] = SystemResult(means, per_query, clusters)
    run.stats.update({"num_queries": len(data.queries), "num_docs": len(data.corpus)})


# ------------------------------------------------------------------------------ localization tasks


class SnapshotLoader:
    """Build lexical CodeIndexes for (repo, commit) snapshots, reusing work across nearby commits.

    Parsed chunks are cached per (path, blob id) and BM25 term counts per chunk text, so a snapshot only
    re-reads and re-chunks the files that changed since an earlier snapshot of the same repository.
    Caches are dropped when the repository changes.
    """

    def __init__(self, repos_dir: Path, chunker: Chunker, languages=DEFAULT_LANGUAGES):
        self.repos_dir, self.chunker, self.languages = repos_dir, chunker, languages
        self._repo: str | None = None
        self._blob_chunks: dict[tuple[str, str], list[Chunk]] = {}
        self._term_arrays: dict[str, tuple] = {}
        self._vocab: dict[str, int] = {}

    def _switch(self, repo: str) -> None:
        if repo != self._repo:
            self._repo = repo
            self._blob_chunks.clear()
            self._term_arrays.clear()
            self._vocab = {}

    def load(self, repo: str, commit: str | None) -> CodeIndex:
        self._switch(repo)
        path = self.repos_dir / repo
        if commit is None:
            chunks, hashes = chunk_files(iter_source_files(path, self.languages), self.chunker)
            return CodeIndex(chunks=chunks, manifest={"files": hashes})
        blobs = list_source_blobs(path, commit, self.languages)
        missing = [b for b in blobs if b not in self._blob_chunks]
        if missing:  # no `git cat-file` process when every blob was already chunked for an earlier snapshot
            for file_path, obj, source in read_source_blobs(path, missing):
                self._blob_chunks[(file_path, obj)] = self.chunker.chunk(source) if source is not None else []
        chunks = [c for b in blobs for c in self._blob_chunks[b]]
        indexed = {p: obj for (p, obj) in blobs if self._blob_chunks[(p, obj)]}
        return CodeIndex(chunks=chunks, manifest={"files": indexed})

    def bm25(self, index: CodeIndex) -> BM25Index:
        arrays = []
        for c in index.chunks:
            arr = self._term_arrays.get(c.text)
            if arr is None:
                arr = self._term_arrays[c.text] = term_arrays(Counter(code_tokens(c.text)), self._vocab)
            arrays.append(arr)
        return BM25Index(vocab=self._vocab, arrays=arrays)


def tag_groups(cases: Sequence[LocalizationCase]) -> dict[str, set[str]]:
    """Case ids per tag group: each `key=value`, plus the full combination when cases carry several tags."""
    groups: dict[str, set[str]] = defaultdict(set)
    for case in cases:
        items = sorted(case.tags.items())
        for k, v in items:
            groups[f"{k}={v}"].add(case.id)
        if len(items) > 1:
            groups[",".join(f"{k}={v}" for k, v in items)].add(case.id)
    return dict(sorted(groups.items()))


def _subset(result: SystemResult, ids: set[str]) -> SystemResult | None:
    per_query = {q: v for q, v in result.per_query.items() if q in ids}
    if not per_query:
        return None
    names = next(iter(per_query.values())).keys()
    metrics = {m: sum(pq[m] for pq in per_query.values()) / len(per_query) for m in names}
    metrics["num_queries"] = len(per_query)
    return SystemResult(metrics, per_query, {q: c for q, c in result.clusters.items() if q in per_query})


CHECKPOINT_TOP = 100  # ranks kept per query in per-repository checkpoints (metrics use k <= 50)


def _checkpoint_key(repo_cases: Sequence[LocalizationCase], systems, variant: str = "") -> str:
    ident = [
        [(c.id, c.commit, c.query, c.gold_files, c.gold_symbols) for c in repo_cases],
        [(name, mode, getattr(m, "identity", getattr(m, "fingerprint", None))) for name, m, mode in systems],
        LOCALIZATION_DEPTH,
    ]
    if variant:  # omitted for default settings, so their key matches a call without `variant`
        ident.append(variant)
    return hashlib.sha256(json.dumps(ident, sort_keys=True, default=str).encode()).hexdigest()


def _score_repo(
    repo: str,
    groups: list[tuple[str | None, list[LocalizationCase]]],
    loader: SnapshotLoader,
    systems,
    caches,
    run: TaskRun,
    query_transform: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Rankings and filtered gold for one repository's cases (all its snapshots)."""
    memos = {id(m): {} for _, m, _ in systems if m is not None}  # chunk text -> vector, for this repository
    out: dict[str, Any] = {
        "file_qrels": {}, "symbol_qrels": {}, "file_rank": defaultdict(dict), "sym_rank": defaultdict(dict),
        "missing_gold": 0, "skipped": 0, "chunks": 0, "snapshots": len(groups),
    }
    for commit, group in groups:
        index = loader.load(repo, commit)
        out["chunks"] += len(index.chunks)
        indexed = set(index.manifest["files"])
        symbols = {f"{c.path}::{c.symbol}" for c in index.chunks if c.symbol}
        scored = []
        for case in group:
            # Gold files absent from the snapshot (non-code, unsupported language, generated) cannot be
            # retrieved by any system; they are dropped and counted.
            gold = [f for f in case.gold_files if f in indexed]
            out["missing_gold"] += len(case.gold_files) - len(gold)
            if not gold:
                out["skipped"] += 1
                continue
            out["file_qrels"][case.id] = dict.fromkeys(gold, 1.0)
            gold_syms = [s for s in case.gold_symbols if s in symbols]
            if gold_syms:
                out["symbol_qrels"][case.id] = dict.fromkeys(gold_syms, 1.0)
            scored.append(case)
        if not scored:
            continue
        queries = [query_transform(c.query) if query_transform else c.query for c in scored]
        bm25: BM25Index | None = None
        embeddings: dict[int, object] = {}
        for name, model, mode in systems:
            emb = None
            if model is not None:
                if id(model) not in embeddings:
                    t = time.perf_counter()
                    embeddings[id(model)] = encode_documents_cached(model, [c.text for c in index.chunks], caches[id(model)], memo=memos[id(model)])
                    timing = run.timings[model.name]
                    timing["encode_chunks_seconds"] = timing.get("encode_chunks_seconds", 0.0) + time.perf_counter() - t
                emb = embeddings[id(model)]
            if split_mode(mode)[0] in ("bm25", "hybrid") and bm25 is None:
                t = time.perf_counter()
                bm25 = loader.bm25(index)
                run.timings["bm25"]["build_seconds"] = run.timings["bm25"].get("build_seconds", 0.0) + time.perf_counter() - t
            dense_weight = split_mode(mode)[1] or 1.0  # "hybrid@2.0": dense weight in the RRF, BM25 stays 1
            retriever = Retriever(
                CodeIndex(chunks=index.chunks, embeddings=emb), model, candidates=LOCALIZATION_DEPTH, bm25=bm25,
                dense_weight=dense_weight,
            )
            hits = retriever.search_many(queries, k=LOCALIZATION_DEPTH, mode=mode)
            for case, h in zip(scored, hits):
                out["file_rank"][name][case.id] = [x.key for x in aggregate(h, index.chunks, file_key)][:CHECKPOINT_TOP]
                out["sym_rank"][name][case.id] = [x.key for x in aggregate(h, index.chunks, symbol_key)][:CHECKPOINT_TOP]
    return out


def run_localization_task(
    cases: Sequence[LocalizationCase],
    repos_dir: Path,
    chunker: Chunker,
    systems: Sequence[tuple[str, EmbeddingModel | None, str]],
    ks: Sequence[int],
    cache_dir: str | None,
    run: TaskRun,
    checkpoint_dir: Path | None = None,
    on_repo_done: Callable[[str, dict[str, Any]], None] | None = None,
    query_transform: Callable[[str], str] | None = None,
    variant: str = "",
) -> None:
    """Score file- and symbol-level localization.

    Snapshots are processed one at a time and scored by every system before being discarded, so memory
    stays flat no matter how many commits a task spans; chunk embeddings are reused through the cache.
    With `checkpoint_dir`, each repository's rankings are saved when it finishes and reused on a rerun
    (keyed by its cases, the systems and model identities), so a crash only loses the current repository.
    `on_repo_done(repo, checkpoint)` is called with each repository's checkpoint (including its key) as soon as
    it is scored or resumed, e.g. to stream results off a short-lived machine.
    """
    by_repo: dict[str, dict[str | None, list[LocalizationCase]]] = defaultdict(lambda: defaultdict(list))
    # Repository-then-time order keeps consecutive snapshots close, maximising blob and embedding reuse.
    for case in sorted(cases, key=lambda c: (c.repo, c.created_at, c.id)):
        by_repo[case.repo][case.commit].append(case)
    loader = SnapshotLoader(repos_dir, chunker)
    caches = {
        id(m): EmbeddingCache(cache_dir, m.fingerprint, m.dim) if cache_dir else None
        for _, m, _ in systems
        if m is not None
    }
    file_qrels: dict[str, dict[str, float]] = {}
    symbol_qrels: dict[str, dict[str, float]] = {}
    file_rank: dict[str, dict[str, list[str]]] = defaultdict(dict)
    sym_rank: dict[str, dict[str, list[str]]] = defaultdict(dict)
    missing_gold = skipped = total_chunks = n_snapshots = resumed = 0
    for repo, groups in by_repo.items():
        repo_cases = [c for g in groups.values() for c in g]
        key = _checkpoint_key(repo_cases, systems, variant)
        path = checkpoint_dir / f"{repo}.json" if checkpoint_dir else None
        part = None
        if path is not None and path.exists():
            saved = json.loads(path.read_text())
            if saved.get("key") == key:
                part, resumed = saved, resumed + 1
        if part is None:
            logger.info("localization: %s (%d cases, %d snapshots)", repo, len(repo_cases), len(groups))
            part = _score_repo(repo, list(groups.items()), loader, systems, caches, run, query_transform)
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps({"key": key, **part}))
                tmp.replace(path)
        if on_repo_done is not None:
            on_repo_done(repo, {"key": key, **part})
        file_qrels.update(part["file_qrels"])
        symbol_qrels.update(part["symbol_qrels"])
        for name in part["file_rank"]:
            file_rank[name].update(part["file_rank"][name])
            sym_rank[name].update(part["sym_rank"][name])
        missing_gold += part["missing_gold"]
        skipped += part["skipped"]
        total_chunks += part["chunks"]
        n_snapshots += part["snapshots"]
    clusters = {c.id: c.repo for c in cases}
    groups_by_tag = tag_groups(cases)
    for name, _, _ in systems:
        levels = {"symbol": (sym_rank, symbol_qrels), "file": (file_rank, file_qrels)}
        for level, (ranks, qrels) in levels.items():
            if not qrels:
                continue
            means, per_query = evaluate_rankings(ranks[name], qrels, ks)
            result = SystemResult(means, per_query, {q: clusters[q] for q in per_query})
            run.results[level][name] = result
            for group, ids in groups_by_tag.items():
                sub = _subset(result, ids)
                if sub is not None:
                    run.results[f"{level}[{group}]"][name] = sub
    for cache in caches.values():
        if cache is not None:
            cache.close()
    scored = [c for c in cases if c.id in file_qrels]
    ceilings = [c.meta["symbol_recall_ceiling"] for c in scored if "symbol_recall_ceiling" in c.meta]
    run.stats.update(
        {
            "num_cases": len(cases),
            "scored_cases": len(file_qrels),
            "repos": len(by_repo),
            "skipped_cases_no_indexed_gold": skipped,
            "gold_files_not_indexed": missing_gold,
            "cases_with_symbol_gold": len(symbol_qrels),
            **{f"cases[{g}]": len(ids & file_qrels.keys()) for g, ids in groups_by_tag.items()},
            "snapshots": n_snapshots,
            "mean_chunks_per_snapshot": round(total_chunks / max(1, n_snapshots), 1),
            "repos_resumed_from_checkpoint": resumed,
        }
    )
    if ceilings:
        # Gold created by the PR does not exist at its base commit: report how much recall is out of reach.
        run.stats.update(
            {
                "cases_with_unreachable_symbols": sum(1 for x in ceilings if x < 1.0),
                "cases_symbol_level_unreachable": sum(1 for c in scored if not c.gold_symbols),
                "mean_symbol_recall_ceiling": round(sum(ceilings) / len(ceilings), 4),
                "mean_file_recall_ceiling": round(
                    sum(c.meta.get("file_recall_ceiling", 1.0) for c in scored) / max(1, len(scored)), 4
                ),
            }
        )


# ----------------------------------------------------------------------------------------- driver


def _environment() -> dict[str, Any]:
    import torch
    import transformers

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "", False
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "codeintel_commit": commit,
        "codeintel_dirty": dirty,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def _contamination_note(task: TaskConfig) -> str:
    if task.contamination_note:
        return KNOWN_EXPOSURE.get(task.contamination_note, task.contamination_note)
    return ""


def run_evaluation(
    cfg: EvaluationConfig,
    model_loader: Callable[[str], EmbeddingModel] | None = None,
    output_dir: str | Path | None = None,
    on_repo_done: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Evaluate every model in `cfg` on every task; writes results.json, per_query.jsonl, report.md."""
    from codeintel.models.base import load_model

    loader = model_loader or (lambda path: load_model(load_model_config(path)))
    out_dir = Path(output_dir or cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chunker = Chunker(ChunkerConfig(context=cfg.chunk_context))
    query_transform = None
    if cfg.query_cleaning:
        from functools import partial

        from codeintel.retrieval.query import clean_issue_query

        query_transform = partial(clean_issue_query, max_chars=cfg.query_max_chars)
    runs: dict[str, TaskRun] = {t.name: TaskRun() for t in cfg.tasks}
    # The models are small, so they are loaded together; each task's data (and each repository snapshot)
    # is then prepared once and scored by every system.
    models = [loader(path) for path in cfg.models]
    model_names = [m.name for m in models]
    if len(set(model_names)) != len(model_names):
        raise ValueError(f"model names must be unique, got {model_names}")
    systems: list[tuple[str, EmbeddingModel | None, str]] = []
    if "bm25" in cfg.retrieval_modes:
        systems.append(("bm25", None, "bm25"))
    for model in models:
        systems += [(f"{model.name}/{mode}", model, mode) for mode in cfg.retrieval_modes if mode != "bm25"]
    for task in cfg.tasks:
        logger.info("task %s: %s", task.name, [s[0] for s in systems])
        if task.type == "retrieval":
            data = load_retrieval_task(task, cfg.seed)
            run_retrieval_task(data, systems, cfg.ks, cfg.cache_dir, runs[task.name])
        elif task.type == "localization":
            cases = read_records(task.path, LocalizationCase)
            if task.max_queries and len(cases) > task.max_queries:
                cases = sorted(random.Random(cfg.seed).sample(cases, task.max_queries), key=lambda c: c.id)
            repos_dir = Path(task.repos_dir or Path(task.path).parent / "repos")
            run_localization_task(
                cases, repos_dir, chunker, systems, cfg.ks, cfg.cache_dir, runs[task.name],
                checkpoint_dir=out_dir / "checkpoints" / task.name, on_repo_done=on_repo_done,
                query_transform=query_transform, variant=cfg.variant,
            )
        else:
            raise ValueError(f"unknown task type {task.type!r}")
    del models
    gc.collect()

    results = _summarize(cfg, runs, model_names)
    results["environment"] = _environment()
    results["models"] = cfg.models
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    with open(out_dir / "per_query.jsonl", "w") as f:
        for task_name, run in runs.items():
            for level, systems in run.results.items():
                for system, res in systems.items():
                    for qid, vals in res.per_query.items():
                        f.write(json.dumps({"task": task_name, "level": level, "system": system, "query": qid, **vals}) + "\n")
    (out_dir / "report.md").write_text(render_report(results))
    return results


def _summarize(cfg: EvaluationConfig, runs: dict[str, TaskRun], model_names: list[str]) -> dict[str, Any]:
    cfg_modes = list(cfg.retrieval_modes)
    primary = cfg.primary_metric
    out: dict[str, Any] = {"primary_metric": primary, "tasks": {}}
    for task in cfg.tasks:
        run = runs[task.name]
        task_out: dict[str, Any] = {
            "type": task.type,
            "stats": run.stats,
            "timings": {k: dict(v) for k, v in run.timings.items()},
            "contamination": _contamination_note(task),
            "levels": {},
        }
        for level, systems in run.results.items():
            level_out: dict[str, Any] = {"systems": {}, "comparisons": []}
            for system, res in systems.items():
                qids = list(res.per_query)
                values = [res.per_query[q][primary] for q in qids]
                entry: dict[str, Any] = {"metrics": res.metrics}
                if res.clusters and len(set(res.clusters.values())) >= MIN_CLUSTERS:
                    groups = [res.clusters[q] for q in qids]
                    entry[f"{primary}_ci95"] = cluster_bootstrap_ci(values, groups, cfg.bootstrap_samples, cfg.seed)
                    entry["macro"] = {
                        m: macro_mean([res.per_query[q][m] for q in qids], groups)
                        for m in (primary, "recall@10", "recall@50")
                        if qids and m in res.per_query[qids[0]]
                    }
                    entry[f"macro_{primary}_ci95"] = cluster_bootstrap_ci(
                        values, groups, cfg.bootstrap_samples, cfg.seed, macro=True
                    )
                    entry["num_repos"] = len(set(groups))
                else:
                    entry[f"{primary}_ci95"] = bootstrap_ci(values, cfg.bootstrap_samples, cfg.seed)
                level_out["systems"][system] = entry
            pairs = []
            for mode in [m for m in cfg_modes if m != "bm25"]:
                names = [f"{m}/{mode}" for m in model_names if f"{m}/{mode}" in systems]
                pairs += [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
                if "bm25" in systems:
                    pairs += [(n, "bm25") for n in names]
            for a_name, b_name in pairs:
                a, b = systems[a_name], systems[b_name]
                qids = sorted(set(a.per_query) & set(b.per_query))
                xs = [a.per_query[q][primary] for q in qids]
                ys = [b.per_query[q][primary] for q in qids]
                cmp = {"a": a_name, "b": b_name, "metric": primary}
                if a.clusters and len({a.clusters[q] for q in qids}) >= MIN_CLUSTERS:
                    cmp.update(clustered_paired_comparison(xs, ys, [a.clusters[q] for q in qids], seed=cfg.seed))
                else:
                    cmp.update(paired_comparison(xs, ys, seed=cfg.seed))
                level_out["comparisons"].append(cmp)
            task_out["levels"][level] = level_out
        out["tasks"][task.name] = task_out
    return out


REPORT_METRICS = {
    "doc": ["ndcg@10", "mrr", "recall@10", "hit@1", "recall@20"],
    "symbol": ["ndcg@10", "recall@5", "recall@10", "recall@20", "recall@50", "hit@1", "mrr"],
    "file": ["ndcg@10", "recall@5", "recall@10", "hit@1", "hit@5", "full_recall@10", "mrr"],
}
LEVEL_ORDER = {"symbol": 0, "file": 1, "doc": 2}


def _base_level(level: str) -> str:
    return level.split("[", 1)[0]


def _level_sort_key(level: str) -> tuple[int, int, str]:
    return (LEVEL_ORDER.get(_base_level(level), 9), "[" in level, level)


def _fmt_cmp(c: dict) -> str:
    text = f"{c['a']} vs {c['b']}: {c['mean_diff']:+.4f} [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]"
    if "macro_diff" in c:
        text += f"; macro {c['macro_diff']:+.4f} [{c['macro_ci_low']:+.4f}, {c['macro_ci_high']:+.4f}]"
        return text + f", p={c['p_value']:.4f} (repo-level), n={c['n']}, repos={c['n_clusters']}"
    return text + f", p={c['p_value']:.4f}, n={c['n']}"


def _breakdown_table(levels: dict[str, Any], base: str, primary: str) -> list[str]:
    """Systems x tag groups for one level, primary metric and recall@10 per cell."""
    groups = [lv for lv in sorted(levels, key=_level_sort_key) if _base_level(lv) == base and "[" in lv]
    if not groups:
        return []
    names = [g[len(base) + 1 : -1] for g in groups]
    systems = list(levels[base]["systems"])
    lines = [f"Breakdown ({base}-level {primary} / recall@10, n per group in header):", ""]
    header = ["system", "all"] + [
        f"{n} (n={levels[g]['systems'][systems[0]]['metrics'].get('num_queries', 0):.0f})" for n, g in zip(names, groups)
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|---" * len(header) + "|")
    for system in systems:
        cells = []
        for lv in [base] + groups:
            m = levels[lv]["systems"].get(system, {}).get("metrics", {})
            cells.append(f"{m.get(primary, 0.0):.3f} / {m.get('recall@10', 0.0):.3f}")
        lines.append(f"| {system} | " + " | ".join(cells) + " |")
    lines.append("")
    # Per group: models against each other in the same mode, and each dense model against BM25.
    cmp_lines = [
        f"- [{n}] {_fmt_cmp(c)}"
        for n, g in zip(names, groups)
        for c in levels[g]["comparisons"]
        if c["b"] != "bm25" or c["a"].endswith("/dense")
    ]
    if cmp_lines:
        lines += [f"Paired comparisons per group on {primary}:", ""] + cmp_lines + [""]
    return lines


def render_report(results: dict[str, Any]) -> str:
    primary = results["primary_metric"]
    lines = ["# Evaluation report", ""]
    env = results.get("environment", {})
    if env:
        lines.append(
            f"torch {env.get('torch')} · transformers {env.get('transformers')} · {env.get('platform')} · "
            f"codeintel {env.get('codeintel_commit', '')[:10]}{' (dirty)' if env.get('codeintel_dirty') else ''} · "
            f"{env.get('timestamp')}"
        )
        lines.append("")
    for task_name, task in results["tasks"].items():
        lines += [f"## {task_name} ({task['type']})", ""]
        stats = ", ".join(f"{k}={v}" for k, v in task["stats"].items())
        lines += [f"Stats: {stats}", ""]
        if task.get("contamination"):
            lines += [f"> Contamination: {task['contamination']}", ""]
        levels = task["levels"]
        for level in sorted((lv for lv in levels if "[" not in lv), key=_level_sort_key):
            level_out = levels[level]
            metrics = REPORT_METRICS.get(level, [primary])
            if primary not in metrics:
                metrics = [primary] + metrics
            lines += [f"### {level}-level", ""]
            clustered = any("macro" in s for s in level_out["systems"].values())
            extra = [f"{primary} 95% CI" + (" (repo-clustered)" if clustered else " (case-level)")]
            if clustered:
                extra += [f"macro {primary}", "macro 95% CI", "macro recall@10"]
            lines.append("| system | " + " | ".join(metrics + extra) + " |")
            lines.append("|---" * (len(metrics) + len(extra) + 1) + "|")
            for system, s in level_out["systems"].items():
                m = s["metrics"]
                lo, hi = s[f"{primary}_ci95"]
                row = [f"{m.get(x, 0.0):.4f}" for x in metrics] + [f"[{lo:.4f}, {hi:.4f}]"]
                if clustered:
                    mlo, mhi = s[f"macro_{primary}_ci95"]
                    row += [f"{s['macro'][primary]:.4f}", f"[{mlo:.4f}, {mhi:.4f}]", f"{s['macro'].get('recall@10', 0.0):.4f}"]
                lines.append(f"| {system} | " + " | ".join(row) + " |")
            lines.append("")
            if level_out["comparisons"]:
                lines.append(f"Paired comparisons on {primary} (A − B, bootstrap 95% CI, sign-flip p):")
                lines.append("")
                lines += [f"- {_fmt_cmp(c)}" for c in level_out["comparisons"]]
                lines.append("")
            lines += _breakdown_table(levels, level, primary)
        if task.get("timings"):
            lines.append("Timings:")
            lines.append("")
            for system, t in task["timings"].items():
                lines.append(f"- {system}: " + ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in t.items()))
            lines.append("")
    return "\n".join(lines)
