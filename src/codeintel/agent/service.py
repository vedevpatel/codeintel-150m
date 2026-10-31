"""Semantic code search over one local repository, for coding agents (served over MCP by codeintel.agent.server).

The index lives outside the repository (`~/.cache/codeintel`, or $CODEINTEL_CACHE) and is refreshed whenever a
source file's size or modification time changes. Refreshing re-chunks the repository but only embeds chunks whose
text is new, so after the first build an update costs seconds rather than minutes.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from codeintel.indexing.repository import (
    DEFAULT_LANGUAGES,
    _git_files,
    _skipped_dir,
    _walk_files,
    detect_language,
)
from codeintel.indexing.storage import CodeIndex, EmbeddingCache, build_index
from codeintel.retrieval.hybrid import Retriever, file_key, symbol_key
from codeintel.retrieval.query import clean_issue_query

DEFAULT_MODEL = "vedevpatel/CodeIntel-150M"
MAX_LENGTH = 1024  # the document length the benchmarks were scored at
HYBRID_MODE = "hybrid@2"  # RRF of dense and BM25 with dense weight 2, chosen on the development split
CANDIDATES = 200  # chunk hits collapsed into symbols or files
SNIPPET_LINES = 12
_RELEASE_FILES = ["codeintel_adapter.json", "config.json", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json"]


def cache_root() -> Path:
    return Path(os.environ.get("CODEINTEL_CACHE") or Path.home() / ".cache" / "codeintel")


def load_release_model(repo_id: str = DEFAULT_MODEL, backend: str = "torch", device: str = "auto"):
    """The released model from the Hugging Face Hub: PyTorch on any device (`torch`) or INT8 ONNX on CPU (`onnx`)."""
    from huggingface_hub import snapshot_download

    from codeintel.config import ModelConfig
    from codeintel.models.base import load_model

    weights = ["onnx/model_int8.onnx"] if backend == "onnx" else ["model.safetensors"]
    local_dir = snapshot_download(repo_id, allow_patterns=_RELEASE_FILES + weights)
    cfg = ModelConfig(name=repo_id.split("/")[-1], adapter="onnx" if backend == "onnx" else "hf", repo_id=repo_id,
                      local_dir=local_dir, dtype="int8" if backend == "onnx" else "auto", device=device,
                      max_length=MAX_LENGTH)
    return load_model(cfg)


@dataclass
class Match:
    rank: int
    location: str  # path::QualifiedName, or the path for file-level results
    path: str
    start_line: int
    end_line: int
    score: float
    snippet: str


class RepoSearch:
    """Index one repository on first use and answer natural-language queries with ranked code locations."""

    def __init__(self, root: str | Path, model, cache_dir: str | Path | None = None,
                 languages=DEFAULT_LANGUAGES):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise NotADirectoryError(self.root)
        self.model = model
        self.languages = frozenset(languages)
        base = Path(cache_dir) if cache_dir else cache_root()
        self.embedding_dir = base / "embeddings"
        self.index_dir = base / "indexes" / hashlib.sha256(str(self.root).encode()).hexdigest()[:16]
        self._lock = threading.Lock()
        self._index: CodeIndex | None = None
        self._retriever: Retriever | None = None
        self._signature: str | None = None
        self.last_build_seconds: float | None = None

    # -- index -----------------------------------------------------------------------------------
    def _source_paths(self) -> list[str]:
        paths = _git_files(self.root)
        if paths is None:
            paths = list(_walk_files(self.root))
        return sorted(p for p in set(paths) if not _skipped_dir(p) and detect_language(p) in self.languages)

    def signature(self) -> str:
        """A hash of every source file's path, size and modification time (a stat call per file, no reads)."""
        digest = hashlib.sha256()
        for rel in self._source_paths():
            try:
                st = (self.root / rel).stat()
            except OSError:
                continue
            digest.update(f"{rel}\0{st.st_size}\0{st.st_mtime_ns}\n".encode())
        return digest.hexdigest()

    def _load_saved(self, signature: str) -> CodeIndex | None:
        try:
            index = CodeIndex.load(self.index_dir)
        except (OSError, RuntimeError, ValueError):
            return None
        m = index.manifest
        if m.get("signature") == signature and m.get("model_fingerprint") == self.model.fingerprint:
            return index
        return None

    def ensure_index(self, force: bool = False) -> CodeIndex:
        """The current index, rebuilt (incrementally) if any source file changed since it was built."""
        with self._lock:
            signature = self.signature()
            if not force and self._index is not None and signature == self._signature:
                return self._index
            index = None if force else self._load_saved(signature)
            if index is None:
                start = time.time()
                cache = EmbeddingCache(self.embedding_dir, self.model.fingerprint, self.model.dim)
                try:
                    index = build_index(self.root, model=self.model, cache=cache, languages=self.languages)
                finally:
                    cache.close()
                index.manifest["signature"] = signature
                index.save(self.index_dir)
                self.last_build_seconds = round(time.time() - start, 2)
            self._index, self._signature = index, signature
            self._retriever = Retriever(index, self.model)
            return index

    # -- queries ---------------------------------------------------------------------------------
    def search(self, query: str, k: int = 10, level: str = "symbol") -> list[Match]:
        """Top-k symbols (functions, methods, classes) or files for a bug report, issue or plain question."""
        if level not in ("symbol", "file"):
            raise ValueError("level must be 'symbol' or 'file'")
        index = self.ensure_index()
        assert self._retriever is not None
        cleaned = clean_issue_query(query) or query
        hits = self._retriever.search_many([cleaned], k=max(CANDIDATES, k), mode=HYBRID_MODE)[0]
        key = symbol_key if level == "symbol" else file_key
        seen: dict[str, Match] = {}
        for i, score in hits:
            chunk = index.chunks[i]
            location = key(chunk)
            if location in seen:
                continue
            snippet = "\n".join(chunk.code.splitlines()[:SNIPPET_LINES])
            seen[location] = Match(rank=len(seen) + 1, location=location, path=chunk.path, start_line=chunk.start_line,
                                   end_line=chunk.end_line, score=round(float(score), 5), snippet=snippet)
            if len(seen) == k:
                break
        return list(seen.values())

    def status(self) -> dict:
        index = self._index
        return {
            "repository": str(self.root),
            "model": getattr(self.model, "name", None),
            "indexed": index is not None,
            "files": len(index.manifest.get("files", {})) if index else 0,
            "chunks": len(index.chunks) if index else 0,
            "built_at": index.manifest.get("created_at") if index else None,
            "stale": index is not None and self.signature() != self._signature,
            "last_build_seconds": self.last_build_seconds,
        }


def match_dicts(matches: list[Match]) -> list[dict]:
    return [asdict(m) for m in matches]
