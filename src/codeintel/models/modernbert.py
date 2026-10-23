"""Adapter for Alibaba-NLP/gte-modernbert-base (149M, ModernBERT, Apache-2.0 license).

Inference contract (model card + sentence-transformers config at the pinned revision):
  * CLS pooling (`last_hidden_state[:, 0]`; 1_Pooling/config.json: pooling_mode_cls_token).
  * L2-normalised embeddings, cosine similarity.
  * No query or document prompts.
  * Native transformers architecture (>= 4.48); no remote code.
"""

from __future__ import annotations

from codeintel.models.base import AdapterSpec, TransformerEmbedder

GTE_MODERNBERT_SPEC = AdapterSpec(
    pooling="cls",
    normalize=True,
    query_prefix="",
    document_prefix="",
    recommended_max_length=8192,
    requires_remote_code=False,
    # GeGLU down-projection; quantizing it drops cosine similarity vs fp32 from 0.996 to 0.92.
    int8_fp32_nodes=r"/mlp/Wo/",
)


class GteModernBertAdapter(TransformerEmbedder):
    spec = GTE_MODERNBERT_SPEC
