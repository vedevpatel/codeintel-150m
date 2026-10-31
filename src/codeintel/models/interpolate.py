"""Weight-space interpolation between a base checkpoint and its fine-tuned descendant (WiSE-FT).

theta = (1 - alpha) * theta_base + alpha * theta_ft, parameter by parameter in fp32. alpha = 0 is the base model and
alpha = 1 the fine-tuned one. No training is involved: the result is saved as an ordinary local `hf` checkpoint.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from codeintel.config import ModelConfig
from codeintel.models.base import load_model


def blend_state_dicts(base: dict[str, torch.Tensor], tuned: dict[str, torch.Tensor], alpha: float) -> dict[str, torch.Tensor]:
    if set(base) != set(tuned):
        diff = sorted(set(base) ^ set(tuned))
        raise ValueError(f"base and fine-tuned parameter names differ, e.g. {diff[:5]}")
    out = {}
    for name, b in base.items():
        t = tuned[name]
        if b.shape != t.shape:
            raise ValueError(f"shape mismatch for {name}: {tuple(b.shape)} vs {tuple(t.shape)}")
        out[name] = ((1 - alpha) * b.float() + alpha * t.float()) if b.is_floating_point() else t.clone()
    return out


def interpolate_checkpoint(base_cfg: ModelConfig, tuned_cfg: ModelConfig, alpha: float, out_dir: str | Path, half: bool = True) -> Path:
    """Blend two models of the same architecture and save the result under `out_dir` (a loadable `hf` checkpoint)."""
    cpu = {"device": "cpu", "dtype": "float32"}
    base = load_model(replace(base_cfg, **cpu))
    tuned = load_model(replace(tuned_cfg, **cpu))
    blended = blend_state_dicts(base.model.state_dict(), tuned.model.state_dict(), alpha)
    tuned.model.load_state_dict(blended)
    if half:
        tuned.model = tuned.model.half()  # the same fp16 export format as the fine-tuned checkpoints
    tuned.save(out_dir)
    return Path(out_dir)
