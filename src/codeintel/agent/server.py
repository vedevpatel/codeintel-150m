"""MCP server exposing CodeIntel search over one repository to coding agents (Claude Code, Codex, Cursor, ...).

  codeintel mcp [REPO] [--backend torch|onnx] [--device auto|cpu|cuda|mps] [--model HUB_ID]

Runs over stdio only: the agent launches it as a subprocess, and no network port is opened. The model is downloaded
from the Hugging Face Hub once and cached; the repository is indexed in the background at startup.
"""

from __future__ import annotations

import argparse
import logging
import threading
from typing import Literal

from codeintel.agent.service import DEFAULT_MODEL, RepoSearch, load_release_model, match_dicts

INSTRUCTIONS = (
    "Semantic code search over the current repository. Given a bug report, issue, error description or question, "
    "search_code returns the functions, methods and classes most likely to need changing, ranked, with file paths "
    "and line ranges. Use it before grep when the description does not name the code, then read the top results."
)

logger = logging.getLogger(__name__)


def build_server(search: RepoSearch):
    from mcp.server.mcpserver import MCPServer

    server = MCPServer(name="codeintel", instructions=INSTRUCTIONS)

    @server.tool(description=(
        "Find the code a bug report, issue or question is about. Pass the issue text as written (title and body) or "
        "a plain-language description. Returns ranked locations (path::Symbol, line range, score, snippet)."
    ))
    def search_code(query: str, k: int = 10, level: Literal["symbol", "file"] = "symbol") -> list[dict]:
        return match_dicts(search.search(query, k=max(1, min(k, 50)), level=level))

    @server.tool(description="Whether the repository is indexed, how many files and chunks, and whether it is stale.")
    def index_status() -> dict:
        return search.status()

    @server.tool(description="Rebuild the index now. Only changed code is re-embedded; searches also refresh it.")
    def reindex() -> dict:
        search.ensure_index(force=True)
        return search.status()

    return server


def _warm(search: RepoSearch) -> None:
    try:
        index = search.ensure_index()
        logger.info("indexed %s: %d chunks in %ss", search.root, len(index.chunks), search.last_build_seconds)
    except Exception:  # noqa: BLE001 - surfaced again by the first tool call
        logger.exception("background indexing failed")


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("repo", nargs="?", default=".", help="repository to search (default: current directory)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id")
    p.add_argument("--backend", choices=["torch", "onnx"], default="torch",
                   help="torch (default): on --device; onnx: INT8 on CPU, a smaller download (needs the onnx extra)")
    p.add_argument("--device", default="auto", help="torch backend: auto (CUDA, then MPS, then CPU) | cpu | cuda | mps")
    p.add_argument("--cache-dir", help="index and embedding cache (default: ~/.cache/codeintel or $CODEINTEL_CACHE)")
    p.add_argument("--no-warmup", action="store_true", help="index on the first search instead of at startup")


def run(args: argparse.Namespace) -> int:
    try:
        import mcp  # noqa: F401
    except ImportError:
        raise SystemExit("codeintel mcp needs the mcp extra: install codeintel[mcp]") from None
    model = load_release_model(args.model, backend=args.backend, device=args.device)
    search = RepoSearch(args.repo, model, cache_dir=args.cache_dir)
    if not args.no_warmup:
        threading.Thread(target=_warm, args=(search,), daemon=True).start()
    build_server(search).run("stdio")
    return 0
