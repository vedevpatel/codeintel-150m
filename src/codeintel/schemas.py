"""Core runtime types shared by indexing, retrieval, and evaluation."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any


def split_lines(text: str) -> list[str]:
    """Lines numbered the way git, tree-sitter and `ast` number them: split on "\n" only.

    `str.splitlines()` also breaks on form feeds, vertical tabs, U+2028 etc., which would shift every
    subsequent line relative to parser and diff line numbers.
    """
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return [line[:-1] if line.endswith("\r") else line for line in lines]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()


@dataclass(frozen=True)
class SourceFile:
    """A source file discovered in a repository. `path` is repo-relative and POSIX-style."""

    path: str
    language: str
    text: str

    @property
    def sha256(self) -> str:
        return sha256_text(self.text)


@dataclass(frozen=True)
class Symbol:
    """A named definition extracted by a parser. Lines are 1-based and inclusive."""

    name: str
    qualified_name: str
    kind: str  # function | method | class | interface | type | enum | variable | module
    start_line: int
    end_line: int
    signature: str = ""
    docstring: str = ""
    parent: str | None = None
    # 1-based inclusive line range of the docstring / leading doc comment, if any.
    doc_span: tuple[int, int] | None = None


@dataclass(frozen=True)
class Chunk:
    """A unit of retrieval. `text` is what gets embedded; `code` is the raw source span."""

    chunk_id: str
    path: str
    language: str
    start_line: int
    end_line: int
    kind: str
    symbol: str | None
    text: str
    code: str

    @property
    def content_hash(self) -> str:
        return sha256_text(self.text)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chunk:
        return cls(**{k: data[k] for k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class SearchHit:
    key: str
    score: float
    rank: int
    metadata: dict[str, Any] = field(default_factory=dict)
