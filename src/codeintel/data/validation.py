"""Validate dataset files before they are used for training or evaluation."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from codeintel.data.licenses import is_permissive
from codeintel.data.schemas import (
    TRAINING_TASKS,
    CorpusDoc,
    LocalizationCase,
    Qrel,
    QueryRecord,
    RecordError,
    TrainingPair,
    iter_jsonl,
    parse_record,
)
from codeintel.evaluation.contamination import exact_key

_SHA = re.compile(r"^[0-9a-f]{7,40}$")
MAX_TEXT_CHARS = 100_000


@dataclass
class ValidationReport:
    path: str
    kind: str
    records: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, msg: str) -> None:
        if len(self.errors) < 200:
            self.errors.append(msg)

    def warn(self, msg: str) -> None:
        if len(self.warnings) < 200:
            self.warnings.append(msg)

    def summary(self) -> str:
        status = "OK" if self.ok else "FAILED"
        return f"{status} {self.kind} {self.path}: {self.records} records, {len(self.errors)} errors, {len(self.warnings)} warnings"


def _safe_relpath(path: str) -> bool:
    p = PurePosixPath(path)
    return bool(path) and not p.is_absolute() and ".." not in p.parts and "\\" not in path


def validate_training(path: str | Path) -> ValidationReport:
    report = ValidationReport(str(path), "training")
    pairs: Counter[tuple[str, str]] = Counter()
    for i, row in enumerate(iter_jsonl(path), 1):
        where = f"line {i}"
        try:
            rec = parse_record(TrainingPair, row, where)
        except RecordError as exc:
            report.error(str(exc))
            continue
        report.records += 1
        if not rec.query.strip() or not rec.positive.strip():
            report.error(f"{where}: empty query or positive")
            continue
        if rec.task not in TRAINING_TASKS:
            report.error(f"{where}: unknown task {rec.task!r} (expected one of {sorted(TRAINING_TASKS)})")
        if not rec.source:
            report.error(f"{where}: missing source")
        if not is_permissive(rec.license):
            report.error(f"{where}: licence {rec.license!r} is missing or not permissive")
        if len(rec.query) > MAX_TEXT_CHARS or len(rec.positive) > MAX_TEXT_CHARS:
            report.warn(f"{where}: text longer than {MAX_TEXT_CHARS} chars")
        qk, pk = exact_key(rec.query), exact_key(rec.positive)
        if qk == pk:
            report.error(f"{where}: query identical to positive (trivial pair)")
        if any(exact_key(n) == pk for n in rec.negatives):
            report.error(f"{where}: a hard negative is identical to the positive")
        pairs[(qk, pk)] += 1
    dups = sum(c - 1 for c in pairs.values() if c > 1)
    if dups:
        report.warn(f"{dups} duplicate (query, positive) pairs")
    return report


def validate_localization(path: str | Path, repos_dir: str | Path | None = None) -> ValidationReport:
    report = ValidationReport(str(path), "localization")
    ids: Counter[str] = Counter()
    for i, row in enumerate(iter_jsonl(path), 1):
        where = f"line {i}"
        try:
            case = parse_record(LocalizationCase, row, where)
        except RecordError as exc:
            report.error(str(exc))
            continue
        report.records += 1
        ids[case.id] += 1
        if not case.query.strip():
            report.error(f"{where}: empty query")
        if not case.gold_files:
            report.error(f"{where}: no gold files")
        for f in case.gold_files:
            if not _safe_relpath(f):
                report.error(f"{where}: gold file {f!r} must be a relative POSIX path inside the repo")
        for s in case.gold_symbols:
            if "::" not in s or s.split("::", 1)[0] not in case.gold_files:
                report.error(f"{where}: gold symbol {s!r} must be 'path::Name' with path in gold_files")
        if case.commit is not None and not _SHA.match(case.commit):
            report.error(f"{where}: commit {case.commit!r} is not a hex SHA")
        if not _safe_relpath(case.repo):
            report.error(f"{where}: repo {case.repo!r} must be a relative directory name")
        elif repos_dir is not None and not (Path(repos_dir) / case.repo).is_dir():
            report.error(f"{where}: repository {case.repo!r} not found under {repos_dir}")
    for cid, n in ids.items():
        if n > 1:
            report.error(f"duplicate case id {cid!r} ({n}x)")
    return report


def validate_retrieval_dir(directory: str | Path) -> ValidationReport:
    directory = Path(directory)
    report = ValidationReport(str(directory), "retrieval")
    ids: dict[str, set[str]] = {}
    texts: Counter[str] = Counter()
    for name, cls in (("corpus", CorpusDoc), ("queries", QueryRecord)):
        seen: Counter[str] = Counter()
        for i, row in enumerate(iter_jsonl(directory / f"{name}.jsonl"), 1):
            try:
                rec = parse_record(cls, row, f"{name}.jsonl line {i}")
            except RecordError as exc:
                report.error(str(exc))
                continue
            report.records += 1
            seen[rec.id] += 1
            if name == "corpus":
                texts[rec.text] += 1
            if not rec.text.strip():
                report.error(f"{name}.jsonl line {i}: empty text")
        for rid, n in seen.items():
            if n > 1:
                report.error(f"{name}.jsonl: duplicate id {rid!r} ({n}x)")
        ids[name] = set(seen)
    dup_docs = sum(n - 1 for n in texts.values() if n > 1)
    if dup_docs:
        report.error(f"corpus has {dup_docs} documents duplicating another's text (ties make rankings arbitrary)")
    qrels_path = directory / "qrels.jsonl"
    if not qrels_path.exists():
        report.error("missing qrels.jsonl")
        return report
    judged: set[str] = set()
    for i, row in enumerate(iter_jsonl(qrels_path), 1):
        try:
            q = parse_record(Qrel, row, f"qrels.jsonl line {i}")
        except RecordError as exc:
            report.error(str(exc))
            continue
        if q.query_id not in ids["queries"]:
            report.error(f"qrels.jsonl line {i}: unknown query {q.query_id!r}")
        if q.doc_id not in ids["corpus"]:
            report.error(f"qrels.jsonl line {i}: unknown document {q.doc_id!r}")
        if q.relevance > 0:
            judged.add(q.query_id)
    unjudged = ids["queries"] - judged
    if unjudged:
        report.warn(f"{len(unjudged)} queries have no positive judgement and will be skipped")
    return report
