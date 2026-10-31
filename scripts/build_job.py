"""Build a self-contained GPU job script from a template: the package source, configs and (optionally) cases are
embedded as a base64 payload, so the job needs no checkout on the remote machine.

  uv run python scripts/build_job.py                                    # issue-localization evaluation job
  uv run python scripts/build_job.py --template scripts/job_train.py --cases data/dev/localization/cases.jsonl \
      --out artifacts/jobs/train_job.py

Cases are slimmed to the fields the runner reads (recall ceilings kept); checkpoint keys do not depend on the rest, so
rankings produced from slimmed cases resume against the full cases locally.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KEEP_META = ("file_recall_ceiling", "symbol_recall_ceiling")


def build(cases_path: Path | None, out: Path, template_path: Path | None = None, extra_dirs: tuple[str, ...] = ()) -> dict[str, int]:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=9) as tar:
        for path in sorted((ROOT / "src" / "codeintel").rglob("*.py")):
            tar.add(path, arcname=str(path.relative_to(ROOT)))
        for path in sorted((ROOT / "configs").rglob("*.yaml")):
            tar.add(path, arcname=str(path.relative_to(ROOT)))
        for extra in extra_dirs:  # data directories (repo-relative) shipped inside the bundle
            for path in sorted((ROOT / extra).rglob("*")):
                if path.is_file():
                    tar.add(path, arcname=str(path.relative_to(ROOT)))
        n_cases = 0
        if cases_path is not None:
            slim = io.BytesIO()
            for line in cases_path.read_text().splitlines():
                row = json.loads(line)
                row["meta"] = {k: v for k, v in row.get("meta", {}).items() if k in KEEP_META}
                slim.write((json.dumps(row, separators=(",", ":")) + "\n").encode())
                n_cases += 1
            for name, data in (("cases.jsonl", slim.getvalue()), ("chunk_counts.json", (ROOT / "artifacts/chunk_counts.json").read_bytes())):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    payload = base64.b64encode(buf.getvalue()).decode()
    template = (template_path or ROOT / "scripts" / "job_eval.py").read_text()
    marker = 'PAYLOAD = ""'
    assert template.count(marker) == 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(template.replace(marker, f'PAYLOAD = "{payload}"'))
    return {"cases": n_cases, "payload_bytes": len(payload), "script_bytes": out.stat().st_size}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(ROOT / "data/eval/issue-localization/cases.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "artifacts/jobs/eval_job.py"))
    ap.add_argument("--template", help="job template to fill (default: scripts/job_eval.py)")
    ap.add_argument("--no-cases", action="store_true", help="bundle only src/ and configs/ (and --extra-dir data)")
    ap.add_argument("--extra-dir", action="append", default=[], help="repo-relative data directory to include (repeatable)")
    args = ap.parse_args()
    print(build(None if args.no_cases else Path(args.cases), Path(args.out), Path(args.template) if args.template else None,
                tuple(args.extra_dir)))


if __name__ == "__main__":
    main()
