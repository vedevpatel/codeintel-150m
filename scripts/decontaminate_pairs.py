"""Drop training pairs that exactly or nearly duplicate any evaluation or development text.

  uv run python scripts/decontaminate_pairs.py data/train/csn/csn-code2doc.jsonl ... --suffix .clean

Evaluation texts are every `text` and `query` string (20+ characters) in the JSONL files under data/eval and data/dev:
issue texts, retrieval queries and corpora (docstrings, CoSQA, CodeSearchNet, CoIR, code-to-code). A pair is dropped
if its query or positive matches one exactly (normalised) or at MinHash Jaccard >= 0.7
(`codeintel.evaluation.contamination.decontaminate`). Writes <input stem><suffix>.jsonl next to each input.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from codeintel.data.schemas import TrainingPair, read_records, write_jsonl  # noqa: E402
from codeintel.evaluation.contamination import decontaminate  # noqa: E402


def eval_texts() -> list[str]:
    seen: set[str] = set()
    for base in ("data/eval", "data/dev"):
        for path in sorted((ROOT / base).rglob("*.jsonl")):
            with path.open() as f:
                for line in f:
                    row = json.loads(line)
                    for key in ("text", "query"):
                        value = row.get(key)
                        if isinstance(value, str) and len(value) >= 20:
                            seen.add(value)
    return sorted(seen)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--suffix", default=".clean")
    ap.add_argument("--threshold", type=float, default=0.7)
    args = ap.parse_args()
    texts = eval_texts()
    print(f"{len(texts)} evaluation and development texts", flush=True)
    records = {p: list(read_records(p, TrainingPair)) for p in args.inputs}
    # One index over the eval texts, shared by all inputs (decontaminate builds it per call, so batch the inputs).
    flat = [r for rs in records.values() for r in rs]
    kept, stats = decontaminate(flat, texts, threshold=args.threshold)
    keep_ids = {id(r) for r in kept}
    for path, rs in records.items():
        out = Path(path).with_name(Path(path).stem + args.suffix + ".jsonl")
        n = write_jsonl(out, [r for r in rs if id(r) in keep_ids])
        print(f"{path}: {len(rs)} -> {n} ({out})", flush=True)
    print(stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
