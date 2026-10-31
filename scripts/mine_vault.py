"""Docstring -> function and code-context pairs from The Vault (Fsoft-AIC/the-vault-function), licence-filtered.

  uv run python scripts/mine_vault.py --out data/train/vault --python-shards 3 --js-shards 2 --per-language 60000

Only items whose every declared licence passes `is_permissive` are kept; repositories in the eval or dev roles are
excluded, as are test files. Positives use the same `# path::name` header as the mined training chunks (`//` for
JavaScript). Pinned revision; decontaminate the output afterwards (scripts/decontaminate_pairs.py).
"""

from __future__ import annotations

import argparse
import ast
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_dataset import github_slug, load_manifest  # noqa: E402

from codeintel.data.licenses import is_permissive  # noqa: E402
from codeintel.data.mining import is_test_path  # noqa: E402
from codeintel.data.schemas import TrainingPair, write_jsonl  # noqa: E402

REPO, REV = "Fsoft-AIC/the-vault-function", "505c679056e49a2a269b64777ee7c496d22e1440"


def main() -> int:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/train/vault")
    ap.add_argument("--python-shards", type=int, default=3)
    ap.add_argument("--js-shards", type=int, default=2)
    ap.add_argument("--per-language", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    blocked = {github_slug(r["url"]) for r in load_manifest("configs/data/repos.yaml")["repos"] if r["role"] in ("eval", "dev")}
    blocked |= {"gradio-app/gradio", "typeorm/typeorm"}
    stats = Counter()
    doc_pairs, ccr_pairs = [], []
    for lang, n_shards, total, header in (("python", args.python_shards, 33, "#"), ("javascript", args.js_shards, 6, "//")):
        rows = []
        for i in range(n_shards):
            f = hf_hub_download(REPO, f"data/train/full/{lang}-{i:05d}-of-{total:05d}.parquet", repo_type="dataset", revision=REV)
            rows += pq.read_table(f, columns=["repo", "path", "license", "identifier", "short_docstring", "code"]).to_pylist()
        rng = random.Random(f"{args.seed}:{lang}")
        rng.shuffle(rows)
        kept_doc = kept_ccr = 0
        for r in rows:
            stats["seen"] += 1
            try:
                licences = ast.literal_eval(r["license"]) if isinstance(r["license"], str) else list(r["license"] or [])
            except (ValueError, SyntaxError):
                licences = []
            if not licences or not all(is_permissive(x) for x in licences):
                stats["licence"] += 1
                continue
            if r["repo"].lower() in blocked:
                stats["blocked_repo"] += 1
                continue
            if is_test_path(r["path"]):
                stats["test_file"] += 1
                continue
            code, doc = (r["code"] or "").strip("\n"), (r["short_docstring"] or "").strip()
            lines = code.splitlines()
            if not 3 <= len(lines) <= 80:
                stats["length"] += 1
                continue
            text = f"{header} {r['path']}::{r['identifier']}\n" + code
            lic = " AND ".join(sorted(set(licences)))
            meta = {"vault_repo": r["repo"], "vault_rev": REV}
            if len(doc.split()) >= 4 and kept_doc < args.per_language:
                doc_pairs.append(TrainingPair(query=doc, positive=text, task="nl2code", source="vault-docstring",
                                              language=lang, repo=r["repo"], id=f"vault:{r['repo']}:{r['path']}:{r['identifier']}",
                                              license=lic, meta=meta))
                kept_doc += 1
            elif len(lines) >= 8 and kept_ccr < args.per_language // 2:
                cut = max(2, min(len(lines) - 2, round(len(lines) * rng.uniform(0.3, 0.6))))
                rest = "\n".join(lines[cut:])
                if len(rest.strip()) >= 40:
                    ccr_pairs.append(TrainingPair(query=f"{header} {r['path']}::{r['identifier']}\n" + "\n".join(lines[:cut]),
                                                  positive=rest, task="code2code", source="ccr", language=lang, repo=r["repo"],
                                                  id=f"vault-ccr:{r['repo']}:{r['path']}:{r['identifier']}", license=lic, meta=meta))
                    kept_ccr += 1
            if kept_doc >= args.per_language and kept_ccr >= args.per_language // 2:
                break
        print(lang, {"docstring": kept_doc, "ccr": kept_ccr}, flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print({"docstring_pairs": write_jsonl(out / "docstring.jsonl", doc_pairs), "ccr_pairs": write_jsonl(out / "ccr.jsonl", ccr_pairs),
           **stats})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
