"""GPU job step: build weight-interpolation variants (WiSE-FT) of a fine-tuned model and its base.

Run through scripts/job_launch.py after uploading the fine-tuned archive. Embeds src/ and configs/ (scripts/build_job.py).
Environment variables:

  JOB_MODEL    base model name in configs/models
  JOB_FT_TAR   uploaded <model>-ft.tar (default /content/job/<model>-ft.tar; .partNNN pieces are joined)
  JOB_ALPHAS   comma-separated alphas to build, e.g. 0.6,0.8
  JOB_OUT      output directory (default /content/job/sweep)

Writes <JOB_OUT>/<model>-a<alpha>/ and <model>-a<alpha>.yaml for each alpha, plus <model>-ft.yaml (alpha = 1). The base
(alpha = 0) is configs/models/<model>.yaml.
"""

from __future__ import annotations

import base64
import io
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

PAYLOAD = ""  # base64(tar.gz) of src/ and configs/; filled in by build_job.py


def log(msg: str) -> None:
    print(f"[sweep {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    model = os.environ["JOB_MODEL"]
    out = Path(os.environ.get("JOB_OUT", "/content/job/sweep"))
    alphas = [float(a) for a in os.environ.get("JOB_ALPHAS", "0.3,0.4,0.5,0.6,0.7,0.8,0.9").split(",")]
    ft_tar = Path(os.environ.get("JOB_FT_TAR", f"/content/job/{model}-ft.tar"))
    root = Path("/content/job/sweep_bundle")
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD)), mode="r:gz") as tar:
        tar.extractall(root, filter="data")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.48", "einops", "pyyaml", "safetensors",
                    "huggingface-hub", "numpy"], check=True)
    sys.path.insert(0, str(root / "src"))
    os.chdir(root)

    from dataclasses import replace

    from codeintel.config import load_model_config
    from codeintel.models.interpolate import interpolate_checkpoint

    if not ft_tar.exists():  # large uploads arrive in .partNNN pieces
        parts = sorted(ft_tar.parent.glob(f"{ft_tar.name}.part*"))
        assert parts, f"neither {ft_tar} nor its parts are on the VM"
        with open(ft_tar, "wb") as joined:
            for part in parts:
                joined.write(part.read_bytes())
        log(f"joined {len(parts)} parts into {ft_tar} ({ft_tar.stat().st_size / 1e6:.0f} MB)")
    if os.environ.get("JOB_FT_SHA256"):
        import hashlib

        digest = hashlib.sha256(ft_tar.read_bytes()).hexdigest()
        assert digest == os.environ["JOB_FT_SHA256"], f"archive corrupted in transfer: {digest}"
        log("archive checksum verified")
    ft_root = out / "ft"
    shutil.rmtree(ft_root, ignore_errors=True)
    ft_root.mkdir(parents=True)
    with tarfile.open(ft_tar) as tar:
        tar.extractall(ft_root, filter="data")
    ft_dir = ft_root / f"{model}-ft"
    base_cfg = load_model_config(str(root / f"configs/models/{model}.yaml"))
    ft_cfg = replace(load_model_config(str(root / f"configs/models/{model}.yaml")), adapter="hf", local_dir=str(ft_dir), name=f"{model}-ft")

    def write_yaml(name: str, directory: Path) -> Path:
        lines = [f"name: {name}", "adapter: hf", f"repo_id: {base_cfg.repo_id}", f"local_dir: {directory}",
                 f"trust_remote_code: {str(base_cfg.trust_remote_code).lower()}", "dtype: float16", "device: auto"]
        if base_cfg.remote_code_sha256:
            lines += ["remote_code_sha256:"] + [f"  {k}: {v}" for k, v in base_cfg.remote_code_sha256.items()]
        if base_cfg.max_length:
            lines.append(f"max_length: {base_cfg.max_length}")
        path = out / f"{name}.yaml"
        path.write_text("\n".join(lines) + "\n")
        return path

    write_yaml(f"{model}-ft", ft_dir)
    for alpha in alphas:
        name = f"{model}-a{alpha:g}"
        t = time.time()
        directory = out / name
        shutil.rmtree(directory, ignore_errors=True)
        interpolate_checkpoint(base_cfg, ft_cfg, alpha, directory)
        write_yaml(name, directory)
        log(f"{name} built in {time.time() - t:.0f}s")
    if os.environ.get("JOB_TAR"):  # one file to download: <model>-a<alpha>.tar with the weights directory and its YAML
        for a in alphas:
            name = f"{model}-a{a:g}"
            with tarfile.open(out / f"{name}.tar", "w") as tar:
                tar.add(out / f"{name}.yaml", arcname=f"{name}.yaml")
                tar.add(out / name, arcname=name)
            log(f"archive {out / name}.tar ({(out / name).stat().st_size and (out / f'{name}.tar').stat().st_size / 1e6:.0f} MB)")
    print("@@SWEEP_MODELS " + ",".join(str(out / f"{model}-a{a:g}.yaml") for a in alphas))
    return 0


if __name__ == "__main__":
    sys.exit(main())
