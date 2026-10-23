"""Persist chunk indexes and cache embeddings by content hash.

Index directory layout:
    manifest.json    model fingerprint, dimensions, chunker config, per-file hashes
    chunks.jsonl     one Chunk per line, row-aligned with embeddings.npy
    embeddings.npy   float32 (n_chunks, dim), L2-normalised; absent for lexical-only indexes

The embedding cache is a SQLite file per model fingerprint keyed by sha256(chunk text), so re-indexing a
repository (or indexing it at a nearby commit) only encodes chunks whose text changed.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sqlite3
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from codeintel.indexing.chunker import Chunker, ChunkerConfig
from codeintel.indexing.repository import DEFAULT_LANGUAGES, iter_source_files
from codeintel.schemas import Chunk, SourceFile, sha256_text

if TYPE_CHECKING:
    from codeintel.models.base import EmbeddingModel

FORMAT_VERSION = 1


class EmbeddingCache:
    def __init__(self, cache_dir: str | Path, fingerprint: str, dim: int):
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.path = cache_dir / f"{hashlib.sha256(fingerprint.encode()).hexdigest()[:20]}.sqlite"
        self.dim = dim
        self.conn = sqlite3.connect(self.path)
        self.conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB NOT NULL)")
        meta = dict(self.conn.execute("SELECT key, value FROM meta").fetchall())
        if not meta:
            self.conn.execute("INSERT INTO meta VALUES ('fingerprint', ?), ('dim', ?)", (fingerprint, str(dim)))
            self.conn.commit()
        elif meta.get("fingerprint") != fingerprint or meta.get("dim") != str(dim):
            raise RuntimeError(f"cache {self.path} belongs to a different model fingerprint or dimension")

    def get_many(self, keys: Sequence[str]) -> dict[str, np.ndarray]:
        found: dict[str, np.ndarray] = {}
        for i in range(0, len(keys), 500):
            batch = list(keys[i : i + 500])
            placeholders = ",".join("?" * len(batch))
            for key, blob in self.conn.execute(f"SELECT key, vec FROM emb WHERE key IN ({placeholders})", batch):
                found[key] = np.frombuffer(blob, dtype=np.float32)
        return found

    def put_many(self, items: Iterable[tuple[str, np.ndarray]]) -> None:
        self.conn.executemany(
            "INSERT OR REPLACE INTO emb VALUES (?, ?)",
            ((k, np.ascontiguousarray(v, dtype=np.float32).tobytes()) for k, v in items),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


def encode_documents_cached(
    model: EmbeddingModel, texts: Sequence[str], cache: EmbeddingCache | None = None, chunk_size: int = 4096
) -> np.ndarray:
    """Encode documents, reusing cached vectors for identical texts (and deduplicating within the call)."""
    out = np.zeros((len(texts), model.dim), dtype=np.float32)
    keys = [sha256_text(t) for t in texts]
    cached = cache.get_many(sorted(set(keys))) if cache is not None else {}
    todo: dict[str, str] = {}
    for key, text in zip(keys, texts):
        if key not in cached and key not in todo:
            todo[key] = text
    todo_keys = list(todo)
    for i in range(0, len(todo_keys), chunk_size):
        batch_keys = todo_keys[i : i + chunk_size]
        vecs = model.encode_documents([todo[k] for k in batch_keys])
        fresh = dict(zip(batch_keys, vecs))
        cached.update(fresh)
        if cache is not None:
            cache.put_many(fresh.items())
    for row, key in enumerate(keys):
        out[row] = cached[key]
    return out


@dataclass
class CodeIndex:
    chunks: list[Chunk]
    embeddings: np.ndarray | None = None
    manifest: dict = field(default_factory=dict)

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / "chunks.jsonl", "w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        if self.embeddings is not None:
            np.save(directory / "embeddings.npy", self.embeddings.astype(np.float32, copy=False))
        manifest = dict(self.manifest, format_version=FORMAT_VERSION, num_chunks=len(self.chunks))
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))

    @classmethod
    def load(cls, directory: str | Path, mmap: bool = True) -> CodeIndex:
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest.get("format_version") != FORMAT_VERSION:
            raise RuntimeError(f"unsupported index format {manifest.get('format_version')}")
        with open(directory / "chunks.jsonl", encoding="utf-8") as f:
            chunks = [Chunk.from_dict(json.loads(line)) for line in f]
        emb_path = directory / "embeddings.npy"
        embeddings = np.load(emb_path, mmap_mode="r" if mmap else None) if emb_path.exists() else None
        if embeddings is not None and len(embeddings) != len(chunks):
            raise RuntimeError("embeddings and chunks are misaligned")
        return cls(chunks=chunks, embeddings=embeddings, manifest=manifest)


def chunk_files(files: Iterable[SourceFile], chunker: Chunker) -> tuple[list[Chunk], dict[str, str]]:
    chunks: list[Chunk] = []
    hashes: dict[str, str] = {}
    for source in files:
        hashes[source.path] = source.sha256
        chunks.extend(chunker.chunk(source))
    return chunks, hashes


def build_index(
    root: str | Path,
    model: EmbeddingModel | None = None,
    cache: EmbeddingCache | None = None,
    chunker_config: ChunkerConfig | None = None,
    languages: Iterable[str] = DEFAULT_LANGUAGES,
) -> CodeIndex:
    """Chunk a repository and (optionally) embed every chunk. `model=None` builds a lexical-only index."""
    chunker_config = chunker_config or ChunkerConfig()
    chunks, hashes = chunk_files(iter_source_files(root, languages), Chunker(chunker_config))
    embeddings = None
    if model is not None:
        embeddings = encode_documents_cached(model, [c.text for c in chunks], cache)
    manifest = {
        "root": str(Path(root).resolve()),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": getattr(model, "name", None),
        "model_fingerprint": getattr(model, "fingerprint", None),
        "dim": getattr(model, "dim", None),
        "chunker": dataclasses.asdict(chunker_config),
        "files": hashes,
    }
    return CodeIndex(chunks=chunks, embeddings=embeddings, manifest=manifest)
