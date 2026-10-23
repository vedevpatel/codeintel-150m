"""Adapter for nomic-ai/CodeRankEmbed (137M, NomicBERT architecture, MIT license).

Inference contract (model card + sentence-transformers config at the pinned revision):
  * CLS pooling (1_Pooling/config.json: pooling_mode_cls_token).
  * No Normalize module; similarity is cosine, so L2-normalising before dot products is equivalent.
  * Queries MUST be prefixed with "Represent this query for searching relevant code: ".
  * Documents (code) are encoded without a prefix.
  * Architecture is not in transformers, so it requires the repository's remote modeling code.

Reviewed remote code (revision 3c4b60807d71f79b43f3c4363786d9493691f8b1):
  * configuration_hf_nomic_bert.py / modeling_hf_nomic_bert.py are pure PyTorch (rotary + SwiGLU,
    post-norm BERT). The flash-attn / fused flags in config.json are unused by this code.
  * Its custom `from_pretrained` ignores `revision` when fetching weights from the Hub and falls back to
    `torch.load` (pickle) for `pytorch_model.bin`. We therefore always load from a local snapshot that
    contains safetensors only (see SAFE_ALLOW_PATTERNS / verify_snapshot), and because the local-dir
    branch uses `load_state_dict(strict=False)`, every parameter is checked against the checkpoint.
"""

from __future__ import annotations

import types

import torch

from codeintel.models.base import AdapterSpec, TransformerEmbedder

CODERANK_QUERY_PREFIX = "Represent this query for searching relevant code: "

CODERANK_SPEC = AdapterSpec(
    pooling="cls",
    normalize=True,
    query_prefix=CODERANK_QUERY_PREFIX,
    document_prefix="",
    recommended_max_length=8192,
    requires_remote_code=True,
    # SwiGLU down-projection; quantizing it drops cosine similarity vs fp32 from 0.98 to 0.94.
    int8_fp32_nodes=r"/mlp/fc2/",
)


def _extended_attention_mask(self, attention_mask, input_shape, device=None, dtype=None):
    """transformers 4.x `ModuleUtilsMixin.get_extended_attention_mask` for encoder (non-causal) models.

    The pinned remote code calls this method, which transformers 5 removed. Behaviour is identical for the
    2D/3D padding masks the encoder receives: 0 for attended positions, dtype-min for masked ones.
    """
    if dtype is None:
        dtype = next(self.parameters()).dtype
    if attention_mask.dim() == 3:
        extended = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        extended = attention_mask[:, None, None, :]
    else:
        raise ValueError(f"wrong shape for attention_mask: {tuple(attention_mask.shape)}")
    extended = extended.to(dtype=dtype)
    return (1.0 - extended) * torch.finfo(dtype).min


def apply_transformers5_compat(model: torch.nn.Module) -> None:
    """Restore the method the pinned NomicBERT code expects (applied to base and fine-tuned checkpoints)."""
    if not hasattr(model, "get_extended_attention_mask"):
        model.get_extended_attention_mask = types.MethodType(_extended_attention_mask, model)


class CodeRankEmbedAdapter(TransformerEmbedder):
    spec = CODERANK_SPEC
