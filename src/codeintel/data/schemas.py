"""Dataset record types and JSONL I/O.

Formats:
  retrieval task directory  corpus.jsonl (CorpusDoc), queries.jsonl (QueryRecord), qrels.jsonl (Qrel)
  localization task file    cases.jsonl (LocalizationCase)
  training file             *.jsonl (TrainingPair)
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")

TRAINING_TASKS = frozenset(
    {
        "nl2code",  # natural-language description / docstring -> implementation
        "issue2code",  # issue or commit message -> code that changed
        "code2code",  # related implementation, test <-> implementation, call site <-> definition
        "navigation",  # identifier / symbol query -> definition
        "general",  # general-domain text pairs kept to preserve broad usefulness
    }
)


@dataclass
class CorpusDoc:
    id: str
    text: str
    title: str = ""
    path: str = ""
    language: str = ""
    repo: str = ""


@dataclass
class QueryRecord:
    id: str
    text: str
    language: str = ""
    repo: str = ""


@dataclass
class Qrel:
    query_id: str
    doc_id: str
    relevance: float = 1.0


@dataclass
class LocalizationCase:
    id: str
    repo: str  # directory name under the task's repos_dir
    query: str  # issue text / commit message
    gold_files: list[str]
    commit: str | None = None  # index the repository at this commit (the pre-change state)
    gold_symbols: list[str] = field(default_factory=list)  # "path::Qualified.name"
    language: str = ""
    created_at: str = ""
    source: str = ""
    # Stratification labels reported separately by the runner, e.g. {"language": "python", "mentions": "descriptive"}.
    tags: dict[str, str] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingPair:
    query: str
    positive: str
    task: str
    source: str
    negatives: list[str] = field(default_factory=list)
    language: str = ""
    repo: str = ""
    id: str = ""
    license: str = ""  # SPDX expression of the source; validate_training only accepts permissive ones
    meta: dict[str, Any] = field(default_factory=dict)


class RecordError(ValueError):
    pass


def parse_record(cls: type[T], data: dict[str, Any], where: str = "") -> T:
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise RecordError(f"{where}: unknown fields {sorted(unknown)}")
    try:
        return cls(**data)
    except TypeError as exc:
        raise RecordError(f"{where}: {exc}") from exc


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise RecordError(f"{path}:{lineno}: invalid JSON ({exc})") from exc


def read_records(path: str | Path, cls: type[T]) -> list[T]:
    return [parse_record(cls, row, f"{path}:{i}") for i, row in enumerate(iter_jsonl(path), 1)]


def write_jsonl(path: str | Path, records: Iterable[Any]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            row = dataclasses.asdict(r) if dataclasses.is_dataclass(r) else r
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n
