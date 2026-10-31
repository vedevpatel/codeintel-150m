"""Licence-filtered CodeSearchNet training pairs in the formats of CoIR's CodeSearchNet tasks.

  uv run python scripts/mine_csn.py --licences data/raw/csn/licences.jsonl --out data/train/csn --per-language 40000

CodeSearchNet's train split (code-search-net/code_search_net, parquet conversion @ 7e3332b8) is the official training
data for CoIR's CodeSearchNet tasks; its repositories are disjoint from the test split's. A row is used only if the
licence GitHub detects at the row's own commit (scripts/csn_licences.py) passes `is_permissive`. Repositories in the
eval or dev roles and test files are excluded. Three sources, all without our `# path::name` header (CoIR's corpora
have none):

  csn-code2doc  query = the function code (as CoIR's queries), positive = its documentation string
  csn-ccr       query = the code up to a character cut at 40-70% of its length, positive = the rest (as CoIR's CCR)
  csn-docstring query = the documentation string, positive = the code (docstring search)

Decontaminate the output against every evaluation and development text before packing.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from csn_licences import CSN, CSN_REV  # noqa: E402
from prepare_dataset import github_slug, load_manifest  # noqa: E402

from codeintel.data.licenses import is_permissive  # noqa: E402
from codeintel.data.mining import is_test_path  # noqa: E402
from codeintel.data.schemas import TrainingPair, write_jsonl  # noqa: E402


def main() -> int:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    ap = argparse.ArgumentParser()
    ap.add_argument("--licences", default="data/raw/csn/licences.jsonl")
    ap.add_argument("--out", default="data/train/csn")
    ap.add_argument("--per-language", type=int, default=40000, help="pairs per source and language")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    licence = {}
    for r in map(json.loads, Path(args.licences).read_text().splitlines()):
        licence[(r["repo"], r["sha"])] = r["spdx"] if r["status"] == 200 else None
    blocked = {github_slug(r["url"]) for r in load_manifest("configs/data/repos.yaml")["repos"] if r["role"] in ("eval", "dev")}
    blocked |= {"gradio-app/gradio", "typeorm/typeorm"}
    stats = Counter()
    out = {"csn-code2doc": [], "csn-ccr": [], "csn-docstring": []}
    for lang in ("python", "javascript"):
        path = hf_hub_download(CSN, f"{lang}/train/0000.parquet", repo_type="dataset", revision=CSN_REV)
        rows = pq.read_table(path, columns=["repository_name", "func_path_in_repository", "func_name", "func_code_string",
                                            "func_documentation_string", "func_code_url"]).to_pylist()
        rng = random.Random(f"{args.seed}:{lang}")
        rng.shuffle(rows)
        kept = Counter()
        for r in rows:
            stats["seen"] += 1
            repo, sha = r["repository_name"], r["func_code_url"].split("/blob/")[1].split("/")[0]
            if (repo, sha) not in licence:
                stats["licence_unchecked"] += 1
                continue
            spdx = licence[(repo, sha)]
            if not spdx or not is_permissive(spdx):
                stats["licence"] += 1
                continue
            if repo.lower() in blocked:
                stats["blocked_repo"] += 1
                continue
            if is_test_path(r["func_path_in_repository"]):
                stats["test_file"] += 1
                continue
            code, doc = (r["func_code_string"] or "").strip("\n"), (r["func_documentation_string"] or "").strip()
            n_lines = len(code.splitlines())
            if not 3 <= n_lines <= 80 or len(doc.split()) < 3:
                stats["length"] += 1
                continue
            common = dict(language=lang, repo=repo, license=spdx, meta={"csn_rev": CSN_REV, "commit": sha, "path": r["func_path_in_repository"]})
            key = f"{repo}:{r['func_path_in_repository']}:{r['func_name']}"
            # Each function feeds one source, so no text is a positive in two places.
            order = sorted(out, key=lambda s: kept[s])
            for source in order:
                if kept[source] >= args.per_language:
                    continue
                if source == "csn-code2doc":
                    pair = TrainingPair(query=code, positive=doc, task="code2text", source=source, id=f"{source}:{key}", **common)
                elif source == "csn-docstring":
                    pair = TrainingPair(query=doc, positive=code, task="nl2code", source=source, id=f"{source}:{key}", **common)
                else:
                    if len(code) < 120:
                        continue
                    cut = round(len(code) * rng.uniform(0.4, 0.7))
                    pair = TrainingPair(query=code[:cut], positive=code[cut:], task="code2code", source=source, id=f"{source}:{key}", **common)
                out[source].append(pair)
                kept[source] += 1
                break
            if all(kept[s] >= args.per_language for s in out):
                break
        print(lang, dict(kept), flush=True)
    target = Path(args.out)
    target.mkdir(parents=True, exist_ok=True)
    written = {s: write_jsonl(target / f"{s}.jsonl", pairs) for s, pairs in out.items()}
    print({"written": written, **stats})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
