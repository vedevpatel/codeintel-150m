"""GPU job step: contrastive fine-tuning (template for scripts/build_job.py, which embeds the package, configs and
the dev cases). The packed training groups are uploaded separately. Environment variables:

  JOB_MODEL          base model name in configs/models
  JOB_CONFIG         training config name in configs/training
  JOB_STEPS          override max_steps (smoke tests)
  JOB_DATA           packed groups (default /content/job/train_data.json.gz; .partNNN pieces are joined)
  JOB_DATA_SHA256    expected checksum of the packed groups
  JOB_OUT            output directory (default /content/job/run)
  JOB_EXPORT         "final" exports the last checkpoint instead of the best dev one
  JOB_MINI           GradCache mini-batch size (memory and speed only; same gradients)
  JOB_TEACHER        model name in configs/models whose embeddings are precomputed as the distillation teacher
  JOB_TEACHER_CACHE  a teacher-embedding cache computed elsewhere (scripts/teacher_cache.py), used instead

Writes an fp16 export with a model YAML, packed as JOB_OUT/<model>-ft.tar for download.
"""

from __future__ import annotations

import base64
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

PAYLOAD = ""  # base64(tar.gz) of src/, configs/, cases.jsonl (the dev cases); filled in by build_job.py


def log(msg: str) -> None:
    print(f"[train {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    t0 = time.time()
    model = os.environ["JOB_MODEL"]
    config_name = os.environ.get("JOB_CONFIG", "codeintel")
    data = os.environ.get("JOB_DATA", "/content/job/train_data.json.gz")
    out = Path(os.environ.get("JOB_OUT", "/content/job/run"))
    if not Path(data).exists():  # large uploads arrive in .partNNN pieces
        parts = sorted(Path(data).parent.glob(Path(data).name + ".part*"))
        assert parts, f"neither {data} nor its parts are on the VM"
        with open(data, "wb") as joined:
            for part in parts:
                joined.write(part.read_bytes())
        log(f"joined {len(parts)} parts into {data}")
    if os.environ.get("JOB_DATA_SHA256"):
        import hashlib

        assert hashlib.sha256(Path(data).read_bytes()).hexdigest() == os.environ["JOB_DATA_SHA256"], "training data corrupted"
        log("training data checksum verified")
    root = Path("/content/job/train_bundle")
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD)), mode="r:gz") as tar:
        tar.extractall(root, filter="data")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.48", "einops", "tree-sitter>=0.23",
                    "tree-sitter-typescript>=0.23", "pyyaml", "safetensors", "huggingface-hub", "numpy"], check=True)
    sys.path.insert(0, str(root / "src"))
    os.chdir(root)

    import torch
    import yaml

    from codeintel.config import TaskConfig, from_dict
    from codeintel.config import TrainingConfig as TC
    log(f"torch {torch.__version__}; {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU'}")

    manifest = yaml.safe_load((root / "configs/data/repos.yaml").read_text())
    dev_cases = [json.loads(line) for line in (root / "cases.jsonl").read_text().splitlines()]
    dev_repos = sorted({c["repo"] for c in dev_cases})
    by_name = {r["name"]: r for r in manifest["repos"]}
    repos_dir = Path("/content/job/repos")
    repos_dir.mkdir(parents=True, exist_ok=True)
    for name in dev_repos:
        if not (repos_dir / name).exists():
            since = by_name[name].get("shallow_since", manifest.get("shallow_since"))
            subprocess.run(["git", "clone", "--quiet", "--single-branch", f"--shallow-since={since}", by_name[name]["url"], str(repos_dir / name)], check=True)
    log(f"dev repositories ready: {dev_repos}")

    raw = yaml.safe_load((root / f"configs/training/{config_name}.yaml").read_text())
    raw.update({
        "model": str(root / f"configs/models/{model}.yaml"),
        "packed_groups": data,
        "train_files": [],
        "output_dir": str(out),
        "dev_task": {"name": "dev", "type": "localization", "path": str(root / "cases.jsonl"), "repos_dir": str(repos_dir)},
    })
    if os.environ.get("JOB_STEPS"):
        raw["max_steps"] = int(os.environ["JOB_STEPS"])
    if os.environ.get("JOB_MINI"):  # GradCache mini-batch: changes memory and speed, not the gradients
        raw["mini_batch_size"] = int(os.environ["JOB_MINI"])
    if os.environ.get("JOB_TEACHER_CACHE"):  # teacher embeddings precomputed elsewhere (scripts/teacher_cache.py)
        cache_path = Path(os.environ["JOB_TEACHER_CACHE"])
        if not cache_path.exists():  # large uploads arrive in parts
            parts = sorted(cache_path.parent.glob(cache_path.name + ".part*"))
            assert parts, f"neither {cache_path} nor its parts are on the machine"
            with open(cache_path, "wb") as joined:
                for part in parts:
                    joined.write(part.read_bytes())
        if os.environ.get("JOB_TEACHER_CACHE_SHA256"):
            import hashlib

            assert hashlib.sha256(cache_path.read_bytes()).hexdigest() == os.environ["JOB_TEACHER_CACHE_SHA256"], "teacher cache corrupted"
        raw["teacher_cache"] = str(cache_path)
        log(f"using precomputed teacher cache {cache_path} ({cache_path.stat().st_size / 1e9:.2f} GB)")
    elif os.environ.get("JOB_TEACHER"):  # precompute the distillation teacher once (frozen base or another model)
        try:
            from dataclasses import replace as _replace

            from codeintel.config import load_model_config as _lmc
            from codeintel.models.base import load_model as _lm
            from codeintel.training.pack import unpack_groups as _ug
            from codeintel.training.teacher import build_teacher_cache

            t_t = time.time()
            teacher_name = os.environ["JOB_TEACHER"]
            tcfg = _lmc(str(root / f"configs/models/{teacher_name}.yaml"))
            teacher = _lm(_replace(tcfg, max_length=raw.get("max_document_length", 512)))
            gs = _ug(data)
            cache = build_teacher_cache(teacher, [g.query for g in gs], [t for g in gs for t in (*g.positives, *g.negatives)])
            cache_path = Path("/content/job/teacher_cache.pt")
            cache.save(cache_path)
            raw["teacher_cache"] = str(cache_path)
            del teacher, gs
            torch.cuda.empty_cache()
            log(f"teacher cache from {teacher_name}: {len(cache.query_index)} queries, {len(cache.doc_index)} documents "
                f"in {(time.time() - t_t) / 60:.1f} min")
        except Exception as exc:  # noqa: BLE001 - the live teacher computes the same term, only slower
            log(f"teacher cache failed ({type(exc).__name__}: {str(exc)[:200]}); distilling from the live frozen base")
            raw.pop("teacher_cache", None)
    cfg = from_dict(TC, raw, where="train config")
    cfg.dev_task = from_dict(TaskConfig, raw["dev_task"], where="dev_task")
    log(f"training {model}: {cfg.max_steps} steps, batch {cfg.batch_size}, mini-batch {cfg.mini_batch_size}, "
        f"lr {cfg.learning_rate}, shares {cfg.source_shares}")

    from codeintel.training.trainer import Trainer

    trainer = Trainer(cfg)
    log(f"{len(trainer.groups)} training groups; sampler sources {list(trainer.sampler.shares)}")
    final = trainer.train()
    log(f"training done in {(time.time() - t0) / 60:.1f} min; best dev score {trainer.best_score:.4f}")

    # fp16 export of the dev-selected checkpoint, with a model YAML the evaluation pipeline can load.
    # JOB_EXPORT=final exports the last checkpoint; the default keeps the best dev checkpoint.
    best = final if os.environ.get("JOB_EXPORT") == "final" else (out / "best" if (out / "best").exists() else final)
    export = out / "export" / f"{model}-ft"
    export.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(export, ignore_errors=True)
    from copy import deepcopy

    from codeintel.config import load_model_config
    from codeintel.models.base import load_model

    base_cfg = load_model_config(str(root / f"configs/models/{model}.yaml"))
    student = load_model(__import__("dataclasses").replace(base_cfg, adapter="hf", local_dir=str(best), device="cpu", dtype="float32",
                                                           name=f"{model}-ft"))
    student.model = deepcopy(student.model).half()
    student.save(export)
    yaml_lines = [f"name: {model}-ft", "adapter: hf", f"repo_id: {base_cfg.repo_id}", f"local_dir: {export}",
                  f"trust_remote_code: {str(base_cfg.trust_remote_code).lower()}", "dtype: float16", "device: auto"]
    if base_cfg.remote_code_sha256:
        yaml_lines += ["remote_code_sha256:"] + [f"  {k}: {v}" for k, v in base_cfg.remote_code_sha256.items()]
    if base_cfg.max_length:
        yaml_lines.append(f"max_length: {base_cfg.max_length}")
    (export.parent / f"{model}-ft.yaml").write_text("\n".join(yaml_lines) + "\n")
    size = sum(f.stat().st_size for f in export.rglob("*") if f.is_file())
    log(f"exported {export} ({size / 1e6:.0f} MB, fp16) in {(time.time() - t0) / 60:.1f} min total")
    shutil.copy(out / "train_log.jsonl", export.parent / f"{model}-train_log.jsonl")
    archive = out / f"{model}-ft.tar"  # one file to download: weights, tokenizer, adapter spec, model YAML, train log
    with tarfile.open(archive, "w") as tar:
        tar.add(export.parent / f"{model}-ft.yaml", arcname=f"{model}-ft.yaml")
        tar.add(export.parent / f"{model}-train_log.jsonl", arcname=f"{model}-train_log.jsonl")
        tar.add(export, arcname=f"{model}-ft")
    log(f"archive {archive} ({archive.stat().st_size / 1e6:.0f} MB)")
    print("@@TRAIN_DONE " + json.dumps({"model": model, "best_dev": trainer.best_score, "export": str(export), "minutes": round((time.time() - t0) / 60, 1)}))
    return 0


if __name__ == "__main__" or "JOB_MODEL" in os.environ:  # some runners exec the file without setting __name__
    main()
