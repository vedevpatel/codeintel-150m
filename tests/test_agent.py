import asyncio
import os
import shutil
import time

import pytest

from codeintel.agent.service import RepoSearch
from tests.conftest import SAMPLE_REPO


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    shutil.copytree(SAMPLE_REPO, root, ignore=shutil.ignore_patterns("__pycache__"))
    return root


def test_search_returns_ranked_symbols_with_locations(repo, tmp_path, hashing_model):
    search = RepoSearch(repo, hashing_model, cache_dir=tmp_path / "cache")
    matches = search.search("requests fail and are never retried with backoff", k=3)
    assert [m.rank for m in matches] == [1, 2, 3]
    top = matches[0]
    assert top.location == "web/src/api/client.ts::fetchWithRetry" and top.path == "web/src/api/client.ts"
    assert top.start_line <= top.end_line and "fetchWithRetry" in top.snippet
    files = search.search("retry failed requests", k=5, level="file")
    assert len({m.location for m in files}) == len(files) and all("::" not in m.location for m in files)


def test_index_refreshes_only_changed_code_and_persists(repo, tmp_path, hashing_model):
    cache = tmp_path / "cache"
    search = RepoSearch(repo, hashing_model, cache_dir=cache)
    first = search.ensure_index()
    assert search.status()["indexed"] and not search.status()["stale"]
    assert search.ensure_index() is first  # unchanged repository: no rebuild

    encoded = hashing_model.encoded
    target = repo / "inventory" / "pricing.py"
    target.write_text(target.read_text() + "\n\ndef loyalty_points(total):\n    return int(total // 10)\n")
    os.utime(target, ns=(time.time_ns(), time.time_ns()))
    assert search.status()["stale"]
    hits = search.search("how many loyalty points does an order earn", k=1)
    assert hits[0].location == "inventory/pricing.py::loyalty_points"
    assert 0 < hashing_model.encoded - encoded <= 3  # only the new or edited chunks were embedded

    encoded = hashing_model.encoded
    reopened = RepoSearch(repo, hashing_model, cache_dir=cache)
    assert len(reopened.ensure_index().chunks) == len(search.ensure_index().chunks)
    assert hashing_model.encoded == encoded  # loaded from disk, nothing re-embedded


def test_mcp_server_lists_and_calls_tools(repo, tmp_path, hashing_model):
    pytest.importorskip("mcp")
    from codeintel.agent.server import build_server

    server = build_server(RepoSearch(repo, hashing_model, cache_dir=tmp_path / "cache"))
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert names == {"search_code", "index_status", "reindex"}
    result = asyncio.run(server.call_tool("search_code", {"query": "discount rounding is wrong", "k": 2}))
    assert "inventory/pricing.py" in str(result)
