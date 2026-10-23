"""Run retrieval and localization benchmarks for one or more embedding models.

Systems evaluated per task:
  * `bm25`                 lexical baseline (model independent, run once)
  * `<model>/dense`        embedding retrieval with the model's own inference contract
  * `<model>/hybrid`       reciprocal-rank fusion of dense and BM25

Retrieval tasks are scored at document level. Localization tasks index the repository at the case's
commit, retrieve chunks, and score the aggregated file ranking and (when gold symbols exist) the
aggregated symbol ranking.
"""

from __future__ import annotations

import gc
import json
import logging
import platform
import random
import subprocess
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codeintel.config import EvaluationConfig, TaskConfig, load_model_config
from codeintel.data.schemas import CorpusDoc, LocalizationCase, Qrel, QueryRecord, read_records
from codeintel.evaluation.contamination import KNOWN_EXPOSURE
from codeintel.evaluation.metrics import bootstrap_ci, evaluate_rankings, paired_comparison
from codeintel.indexing.chunker import Chunker, ChunkerConfig
from codeintel.indexing.repository import DEFAULT_LANGUAGES, iter_source_files, iter_source_files_at_commit
from codeintel.indexing.storage import CodeIndex, EmbeddingCache, chunk_files, encode_documents_cached
from codeintel.models.base import EmbeddingModel
from codeintel.retrieval.bm25 import BM25Index
from codeintel.retrieval.dense import DenseIndex
from codeintel.retrieval.hybrid import Retriever, aggregate, file_key, reciprocal_rank_fusion, symbol_key

logger = logging.getLogger(__name__)

LOCALIZATION_DEPTH = 1000  # chunks retrieved before aggregating to files / symbols


# ------------------------------------------------------------------------------------------ results


@dataclass
class SystemResult:
    metrics: dict[str, float]
    per_query: dict[str, dict[str, float]]
    ci: dict[str, tuple[float, float]] = field(default_factory=dict)


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
            if mode == "hybrid":
                hits = [
                    reciprocal_rank_fusion([[i for i, _ in d], [i for i, _ in b]])[:depth]
                    for d, b in zip(hits, lexical_hits())
                ]
        rankings = {q.id: [doc_ids[i] for i, _ in h] for q, h in zip(data.queries, hits)}
        means, per_query = evaluate_rankings(rankings, data.qrels, ks)
        run.results["doc"][name] = SystemResult(means, per_query)
    run.stats.update({"num_queries": len(data.queries), "num_docs": len(data.corpus)})


# ------------------------------------------------------------------------------ localization tasks


def load_snapshot(
    repos_dir: Path, repo: str, commit: str | None, chunker: Chunker, languages=DEFAULT_LANGUAGES
) -> CodeIndex:
    """Chunk a repository at `commit` (or its working tree if None) into a lexical-only CodeIndex."""
    path = repos_dir / repo
    files = iter_source_files_at_commit(path, commit, languages) if commit else iter_source_files(path, languages)
    chunks, hashes = chunk_files(files, chunker)
    return CodeIndex(chunks=chunks, manifest={"files": hashes})


def run_localization_task(
    cases: Sequence[LocalizationCase],
    repos_dir: Path,
    chunker: Chunker,
    systems: Sequence[tuple[str, EmbeddingModel | None, str]],
    ks: Sequence[int],
    cache_dir: str | None,
    run: TaskRun,
) -> None:
    """Score file- and symbol-level localization.

    Snapshots are processed one at a time and scored by every system before being discarded, so memory
    stays flat no matter how many commits a task spans; chunk embeddings are reused through the cache.
    """
    groups: dict[tuple[str, str | None], list[LocalizationCase]] = defaultdict(list)
    for case in cases:
        groups[(case.repo, case.commit)].append(case)
    caches = {
        id(m): EmbeddingCache(cache_dir, m.fingerprint, m.dim) if cache_dir else None
        for _, m, _ in systems
        if m is not None
    }
    file_qrels: dict[str, dict[str, float]] = {}
    symbol_qrels: dict[str, dict[str, float]] = {}
    file_rank: dict[str, dict[str, list[str]]] = defaultdict(dict)
    sym_rank: dict[str, dict[str, list[str]]] = defaultdict(dict)
    missing_gold = skipped = total_chunks = 0
    for (repo, commit), group in groups.items():
        index = load_snapshot(repos_dir, repo, commit, chunker)
        total_chunks += len(index.chunks)
        indexed = set(index.manifest["files"])
        symbols = {f"{c.path}::{c.symbol}" for c in index.chunks if c.symbol}
        scored = []
        for case in group:
            # Gold files absent from the snapshot (non-code, unsupported language, generated) cannot be
            # retrieved by any system; they are dropped and counted.
            gold = [f for f in case.gold_files if f in indexed]
            missing_gold += len(case.gold_files) - len(gold)
            if not gold:
                skipped += 1
                continue
            file_qrels[case.id] = dict.fromkeys(gold, 1.0)
            gold_syms = [s for s in case.gold_symbols if s in symbols]
            if gold_syms:
                symbol_qrels[case.id] = dict.fromkeys(gold_syms, 1.0)
            scored.append(case)
        if not scored:
            continue
        queries = [c.query for c in scored]
        bm25: BM25Index | None = None
        embeddings: dict[int, object] = {}
        for name, model, mode in systems:
            emb = None
            if model is not None:
                if id(model) not in embeddings:
                    t = time.perf_counter()
                    embeddings[id(model)] = encode_documents_cached(model, [c.text for c in index.chunks], caches[id(model)])
                    timing = run.timings[model.name]
                    timing["encode_chunks_seconds"] = timing.get("encode_chunks_seconds", 0.0) + time.perf_counter() - t
                emb = embeddings[id(model)]
            if mode in ("bm25", "hybrid") and bm25 is None:
                t = time.perf_counter()
                bm25 = BM25Index([c.text for c in index.chunks])
                run.timings["bm25"]["build_seconds"] = run.timings["bm25"].get("build_seconds", 0.0) + time.perf_counter() - t
            retriever = Retriever(
                CodeIndex(chunks=index.chunks, embeddings=emb), model, candidates=LOCALIZATION_DEPTH, bm25=bm25
            )
            hits = retriever.search_many(queries, k=LOCALIZATION_DEPTH, mode=mode)
            for case, h in zip(scored, hits):
                file_rank[name][case.id] = [x.key for x in aggregate(h, index.chunks, file_key)]
                sym_rank[name][case.id] = [x.key for x in aggregate(h, index.chunks, symbol_key)]
    for name, _, _ in systems:
        means, per_query = evaluate_rankings(file_rank[name], file_qrels, ks)
        run.results["file"][name] = SystemResult(means, per_query)
        if symbol_qrels:
            means, per_query = evaluate_rankings(sym_rank[name], symbol_qrels, ks)
            run.results["symbol"][name] = SystemResult(means, per_query)
    for cache in caches.values():
        if cache is not None:
            cache.close()
    run.stats.update(
        {
            "num_cases": len(cases),
            "scored_cases": len(file_qrels),
            "skipped_cases_no_indexed_gold": skipped,
            "gold_files_not_indexed": missing_gold,
            "cases_with_symbol_gold": len(symbol_qrels),
            "snapshots": len(groups),
            "mean_chunks_per_snapshot": round(total_chunks / max(1, len(groups)), 1),
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
) -> dict[str, Any]:
    """Evaluate every model in `cfg` on every task; writes results.json, per_query.jsonl, report.md."""
    from codeintel.models.base import load_model

    loader = model_loader or (lambda path: load_model(load_model_config(path)))
    out_dir = Path(output_dir or cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chunker = Chunker(ChunkerConfig())
    runs: dict[str, TaskRun] = {t.name: TaskRun() for t in cfg.tasks}
    # Candidate models are small, so they are loaded together; each task's data (and each repository
    # snapshot) is then prepared once and scored by every system.
    models = [loader(path) for path in cfg.models]
    model_names = [m.name for m in models]
    if len(set(model_names)) != len(model_names):
        raise ValueError(f"model names must be unique, got {model_names}")
    systems: list[tuple[str, EmbeddingModel | None, str]] = []
    if "bm25" in cfg.retrieval_modes:
        systems.append(("bm25", None, "bm25"))
    for model in models:
        systems += [(f"{model.name}/{mode}", model, mode) for mode in cfg.retrieval_modes if mode in ("dense", "hybrid")]
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
            run_localization_task(cases, repos_dir, chunker, systems, cfg.ks, cfg.cache_dir, runs[task.name])
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
                values = [pq[primary] for pq in res.per_query.values()] if res.per_query else []
                ci = bootstrap_ci(values, cfg.bootstrap_samples, cfg.seed)
                level_out["systems"][system] = {"metrics": res.metrics, f"{primary}_ci95": ci}
            for mode in ("dense", "hybrid"):
                names = [f"{m}/{mode}" for m in model_names if f"{m}/{mode}" in systems]
                for i in range(len(names)):
                    for j in range(i + 1, len(names)):
                        a, b = systems[names[i]], systems[names[j]]
                        qids = sorted(set(a.per_query) & set(b.per_query))
                        cmp = {"a": names[i], "b": names[j], "metric": primary}
                        cmp.update(
                            paired_comparison(
                                [a.per_query[q][primary] for q in qids],
                                [b.per_query[q][primary] for q in qids],
                                seed=cfg.seed,
                            )
                        )
                        level_out["comparisons"].append(cmp)
            task_out["levels"][level] = level_out
        out["tasks"][task.name] = task_out
    return out


REPORT_METRICS = {
    "doc": ["ndcg@10", "mrr", "recall@10", "hit@1", "recall@20"],
    "file": ["hit@1", "hit@5", "hit@10", "recall@10", "full_recall@10", "mrr", "ndcg@10"],
    "symbol": ["hit@1", "hit@5", "hit@10", "recall@10", "mrr", "ndcg@10"],
}


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
        for level, level_out in task["levels"].items():
            metrics = REPORT_METRICS.get(level, [primary])
            if primary not in metrics:
                metrics = [primary] + metrics
            lines += [f"### {level}-level", ""]
            lines.append("| system | " + " | ".join(metrics) + f" | {primary} 95% CI |")
            lines.append("|---" * (len(metrics) + 2) + "|")
            for system, s in level_out["systems"].items():
                m = s["metrics"]
                lo, hi = s[f"{primary}_ci95"]
                lines.append(
                    f"| {system} | " + " | ".join(f"{m.get(x, 0.0):.4f}" for x in metrics) + f" | [{lo:.4f}, {hi:.4f}] |"
                )
            lines.append("")
            if level_out["comparisons"]:
                lines.append(f"Paired comparisons on {primary} (A − B, bootstrap 95% CI, sign-flip p):")
                lines.append("")
                for c in level_out["comparisons"]:
                    lines.append(
                        f"- {c['a']} vs {c['b']}: {c['mean_diff']:+.4f} [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}], "
                        f"p={c['p_value']:.4f}, n={c['n']}"
                    )
                lines.append("")
        if task.get("timings"):
            lines.append("Timings:")
            lines.append("")
            for system, t in task["timings"].items():
                lines.append(f"- {system}: " + ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in t.items()))
            lines.append("")
    return "\n".join(lines)


def results_to_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for task_name, task in results["tasks"].items():
        for level, level_out in task["levels"].items():
            for system, s in level_out["systems"].items():
                rows.append({"task": task_name, "level": level, "system": system, **s["metrics"]})
    return rows
