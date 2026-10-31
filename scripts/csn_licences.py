"""Licence of each CodeSearchNet training repository AT the commit its functions were taken from (GitHub licence API).

  uv run python scripts/csn_licences.py --top 2000 --out data/raw/csn/licences.jsonl

CodeSearchNet (code-search-net/code_search_net, parquet conversion @ 7e3332b8) mixes licences, so rows are only usable
after a per-repository check. Every function URL pins a commit; GET /repos/{repo}/license?ref={sha} returns the licence
file GitHub detects at that commit (404: no licence file, or the repository is gone). Repositories are taken in order of
function count (the top N per language cover most rows). Resumable: (repo, sha) pairs already in --out are skipped.
The token comes from `gh auth token` and is never printed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CSN, CSN_REV = "code-search-net/code_search_net", "7e3332b8032ff895377dba3e57f0152fcb32ff7d"


def repo_shas(lang: str, top: int) -> list[tuple[str, str, int]]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(CSN, f"{lang}/train/0000.parquet", repo_type="dataset", revision=CSN_REV)
    rows = pq.read_table(path, columns=["repository_name", "func_code_url"]).to_pylist()
    counts = Counter((r["repository_name"], r["func_code_url"].split("/blob/")[1].split("/")[0]) for r in rows)
    return [(repo, sha, n) for (repo, sha), n in counts.most_common(top)]


class GitHub:
    def __init__(self) -> None:
        self.token = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, check=True).stdout.strip()
        self.lock = threading.Lock()
        self.pause_until = 0.0

    def licence(self, repo: str, sha: str) -> dict:
        for attempt in range(5):
            with self.lock:
                wait = self.pause_until - time.time()
            if wait > 0:
                time.sleep(wait)
            req = urllib.request.Request(f"https://api.github.com/repos/{repo}/license?ref={sha}",
                                         headers={"Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json",
                                                  "X-GitHub-Api-Version": "2022-11-28"})
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    self._throttle(resp.headers)
                    d = json.load(resp)
                    lic = d.get("license") or {}
                    return {"status": 200, "spdx": lic.get("spdx_id"), "path": d.get("path"), "full_name": d.get("html_url", "")}
            except urllib.error.HTTPError as e:
                self._throttle(e.headers)
                if e.code == 404:
                    return {"status": 404, "spdx": None}
                if e.code in (403, 429):   # rate limited: _throttle set the pause
                    time.sleep(5)
                    continue
                if e.code >= 500:
                    time.sleep(2 ** attempt)
                    continue
                return {"status": e.code, "spdx": None}
            except (urllib.error.URLError, TimeoutError):
                time.sleep(2 ** attempt)
        return {"status": -1, "spdx": None}

    def _throttle(self, headers) -> None:
        if headers is None:
            return
        remaining, reset = headers.get("X-RateLimit-Remaining"), headers.get("X-RateLimit-Reset")
        if remaining is not None and reset is not None and int(remaining) < 30:
            with self.lock:
                self.pause_until = max(self.pause_until, int(reset) + 5)
            print(f"rate limit nearly used; pausing until {time.strftime('%H:%M', time.localtime(int(reset) + 5))}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=2000, help="repositories per language, by function count")
    ap.add_argument("--languages", default="python,javascript")
    ap.add_argument("--out", default="data/raw/csn/licences.jsonl")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {(r["repo"], r["sha"]) for r in map(json.loads, out.read_text().splitlines())}
    todo = []
    for lang in args.languages.split(","):
        todo += [(lang, repo, sha, n) for repo, sha, n in repo_shas(lang, args.top) if (repo, sha) not in done]
    print(f"{len(done)} already checked; {len(todo)} to check", flush=True)
    gh = GitHub()
    write_lock = threading.Lock()
    stats = Counter()

    def one(item):
        lang, repo, sha, n = item
        res = gh.licence(repo, sha)
        row = {"language": lang, "repo": repo, "sha": sha, "functions": n, **res}
        with write_lock:
            with out.open("a") as f:
                f.write(json.dumps(row) + "\n")
            stats[res["spdx"] or f"status-{res['status']}"] += 1
            if sum(stats.values()) % 250 == 0:
                print(f"{sum(stats.values())}/{len(todo)} {dict(stats.most_common(6))}", flush=True)

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(one, todo))
    print("done", dict(stats.most_common(12)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
