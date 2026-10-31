"""Native PyTorch implementation of the NomicBERT encoder used by CodeRankEmbed (no remote code).

A re-implementation of the inference path of the reviewed `modeling_hf_nomic_bert.py` at CodeRankEmbed revision
3c4b60807d71f79b43f3c4363786d9493691f8b1, for the configuration that checkpoint uses: post-norm blocks, fused QKV
projection without bias, full-width rotary embeddings (GPT-NeoX "rotate half", base taken from the config, fp32
positions), SwiGLU MLP `fc2(fc11(x) * silu(fc12(x)))` without biases, LayerNorm after the embeddings, no learned
position embeddings and no pooler (the embedding is the CLS hidden state). Parameter names equal the checkpoint's,
so `model.safetensors` loads with `strict=True`.

Only that configuration is supported; anything else is rejected rather than silently mis-computed. Inference only:
the reference applies dropout (0.1) in training mode, which this module does not implement, so fine-tuning keeps
using the reviewed reference code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class NomicBertSpec:
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    intermediate_size: int
    type_vocab_size: int
    layer_norm_eps: float
    rotary_base: float

    @classmethod
    def from_config(cls, cfg: dict) -> NomicBertSpec:
        required = {"activation_function": "swiglu", "prenorm": False, "qkv_proj_bias": False, "mlp_fc1_bias": False,
                    "mlp_fc2_bias": False, "rotary_emb_fraction": 1.0, "rotary_emb_interleaved": False, "causal": False,
                    "parallel_block": False, "use_rms_norm": False}
        for key, value in required.items():
            if cfg.get(key) != value:
                raise ValueError(f"native NomicBERT supports {key}={value!r} only, config has {cfg.get(key)!r}")
        if cfg.get("rotary_scaling_factor") or cfg.get("rotary_emb_scale_base"):
            raise ValueError("native NomicBERT does not implement rotary scaling")
        vocab = cfg["vocab_size"]
        multiple = cfg.get("pad_vocab_size_multiple", 1)
        vocab += (multiple - vocab % multiple) % multiple
        return cls(vocab, cfg["n_embd"], cfg["n_layer"], cfg["n_head"], cfg["n_inner"], cfg["type_vocab_size"],
                   cfg["layer_norm_epsilon"], float(cfg["rotary_emb_base"]))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class Attention(nn.Module):
    def __init__(self, spec: NomicBertSpec):
        super().__init__()
        self.num_heads = spec.num_heads
        self.head_dim = spec.hidden_size // spec.num_heads
        self.rotary_base = spec.rotary_base
        self.Wqkv = nn.Linear(spec.hidden_size, 3 * spec.hidden_size, bias=False)
        self.out_proj = nn.Linear(spec.hidden_size, spec.hidden_size, bias=False)

    def rotary(self, seqlen: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = 1.0 / (self.rotary_base ** (torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32) / self.head_dim))
        freqs = torch.outer(torch.arange(seqlen, device=device, dtype=torch.float32), inv_freq)
        cos, sin = torch.cos(freqs).to(dtype), torch.sin(freqs).to(dtype)
        # (seqlen, head_dim/2) -> (1, seqlen, 1, head_dim), first half and second half share frequencies
        return torch.cat([cos, cos], -1)[None, :, None, :], torch.cat([sin, sin], -1)[None, :, None, :]

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        b, s, _ = x.shape
        qkv = self.Wqkv(x).view(b, s, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        cos, sin = self.rotary(s, x.device, x.dtype)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        q, k, v = (t.permute(0, 2, 1, 3) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out_proj(out.permute(0, 2, 1, 3).reshape(b, s, -1))


class Block(nn.Module):
    def __init__(self, spec: NomicBertSpec):
        super().__init__()
        self.attn = Attention(spec)
        self.mlp = nn.Module()
        self.mlp.fc11 = nn.Linear(spec.hidden_size, spec.intermediate_size, bias=False)
        self.mlp.fc12 = nn.Linear(spec.hidden_size, spec.intermediate_size, bias=False)
        self.mlp.fc2 = nn.Linear(spec.intermediate_size, spec.hidden_size, bias=False)
        self.norm1 = nn.LayerNorm(spec.hidden_size, eps=spec.layer_norm_eps)
        self.norm2 = nn.LayerNorm(spec.hidden_size, eps=spec.layer_norm_eps)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        x = self.norm1(self.attn(x, mask) + x)
        h = self.mlp.fc2(self.mlp.fc11(x) * F.silu(self.mlp.fc12(x)))
        return self.norm2(h + x)


class NomicBertEncoder(nn.Module):
    """Returns an object with `last_hidden_state`, like a transformers model, so the adapters can pool it."""

    def __init__(self, spec: NomicBertSpec):
        super().__init__()
        self.spec = spec
        self.config = SimpleNamespace(hidden_size=spec.hidden_size, model_type="nomic_bert_native")
        self.embeddings = nn.Module()
        self.embeddings.word_embeddings = nn.Embedding(spec.vocab_size, spec.hidden_size)
        self.embeddings.token_type_embeddings = nn.Embedding(spec.type_vocab_size, spec.hidden_size)
        self.emb_ln = nn.LayerNorm(spec.hidden_size, eps=spec.layer_norm_eps)
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([Block(spec) for _ in range(spec.num_layers)])

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **_):
        x = self.embeddings.word_embeddings(input_ids) + self.embeddings.token_type_embeddings(torch.zeros_like(input_ids))
        x = self.emb_ln(x)
        mask = None
        if attention_mask is not None:  # additive mask over keys: 0 where attended, dtype-min where padded
            mask = (1.0 - attention_mask[:, None, None, :].to(x.dtype)) * torch.finfo(x.dtype).min
        for layer in self.encoder.layers:
            x = layer(x, mask)
        return SimpleNamespace(last_hidden_state=x)

    @classmethod
    def from_snapshot(cls, directory: str | Path) -> NomicBertEncoder:
        from safetensors.torch import load_file

        directory = Path(directory)
        model = cls(NomicBertSpec.from_config(json.loads((directory / "config.json").read_text())))
        state = {}
        for f in sorted(directory.glob("*.safetensors")):
            state.update(load_file(str(f)))
        state = {k: v for k, v in state.items() if not k.startswith("pooler.")}
        model.load_state_dict({k: v.float() for k, v in state.items()}, strict=True)
        return model
