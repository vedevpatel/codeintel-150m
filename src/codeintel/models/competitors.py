"""Adapters for CodeIntel's base model (granite-r2) and the external models it is compared against.

Each pins the inference contract of its model card / sentence-transformers config at the revision in
configs/models/<name>.yaml. All are native transformers architectures (no remote code) under permissive licences.

  granite-r2       ibm-granite/granite-embedding-english-r2   149M ModernBERT, Apache-2.0, CLS, no prompts
  modernbert-embed nomic-ai/modernbert-embed-base             149M ModernBERT, Apache-2.0, mean pooling,
                                                              "search_query: " / "search_document: " (model card)
  bge-base         BAAI/bge-base-en-v1.5                      110M BERT, MIT, CLS, retrieval query instruction
  qwen3-embed      Qwen/Qwen3-Embedding-0.6B                  596M Qwen3, Apache-2.0, last-token pooling,
                                                              "Instruct: ...\\nQuery:" query prompt; a larger
                                                              reference point, not a ~150M competitor

  jina-v2-code     jinaai/jina-embeddings-v2-base-code        161M JinaBERT (ALiBi), Apache-2.0, mean pooling, no prompts.
                                                              Its modeling code lives in another Hub repository; it is
                                                              vendored at a pinned revision into a local snapshot whose
                                                              auto_map points at the local files (configs/models/jina-v2-code.yaml)
  codesage-small   codesage/codesage-small-v2                 130M, Apache-2.0, mean pooling, EOS appended to every input
                                                              (model card). Reviewed remote code, hash-pinned; weights from
                                                              the SFconvertbot safetensors revision (refs/pr/4)

Not included: snowflake-arctic-embed-m-v2.0 (305M; code in another repository), SFR-Embedding-Code-400M and
jina-code-embeddings-0.5b (non-commercial licences).
"""

from __future__ import annotations

from pathlib import Path

from codeintel.models.base import AdapterSpec, TransformerEmbedder

# One instruction for every task (the model card asks for a task instruction).
QWEN3_CODE_INSTRUCT = "Instruct: Given a code search query or a bug report, retrieve the code that answers or resolves it\nQuery:"


class GraniteR2Adapter(TransformerEmbedder):
    # ModernBERT like gte-modernbert: its GeGLU down-projection is outlier-heavy, so INT8 export keeps it in fp32
    # (quantizing it drops CodeIntel's min cosine vs fp32 to 0.94).
    spec = AdapterSpec(pooling="cls", normalize=True, query_prefix="", document_prefix="", recommended_max_length=8192,
                       int8_fp32_nodes=r"/mlp/Wo/")


class ModernBertEmbedAdapter(TransformerEmbedder):
    spec = AdapterSpec(pooling="mean", normalize=True, query_prefix="search_query: ", document_prefix="search_document: ",
                       recommended_max_length=8192)

    def check_snapshot_contract(self, snapshot: Path) -> None:
        pass  # the prefixes are documented on the model card; the ST config has no prompts


class BgeBaseAdapter(TransformerEmbedder):
    spec = AdapterSpec(pooling="cls", normalize=True, query_prefix="Represent this sentence for searching relevant passages: ",
                       document_prefix="", recommended_max_length=512)

    def check_snapshot_contract(self, snapshot: Path) -> None:
        pass  # the instruction is on the model card; the ST config predates prompts


class Qwen3EmbeddingAdapter(TransformerEmbedder):
    spec = AdapterSpec(pooling="last", normalize=True, query_prefix=QWEN3_CODE_INSTRUCT, document_prefix="",
                       recommended_max_length=8192)
    max_tokens_per_batch = 16384

    def check_snapshot_contract(self, snapshot: Path) -> None:
        pass  # the card asks for a task-specific instruction instead of the default web-search one


class JinaV2CodeAdapter(TransformerEmbedder):
    spec = AdapterSpec(pooling="mean", normalize=True, query_prefix="", document_prefix="", recommended_max_length=8192,
                       requires_remote_code=True)

    def load_backbone(self, cfg):
        import transformers.pytorch_utils as pytorch_utils

        if not hasattr(pytorch_utils, "find_pruneable_heads_and_indices"):
            # Imported by the pinned JinaBERT code, removed in transformers 5; used only for head pruning, never here.
            def find_pruneable_heads_and_indices(*args, **kwargs):
                raise NotImplementedError("attention-head pruning is not supported")

            pytorch_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
        return super().load_backbone(cfg)


class CodeSageAdapter(TransformerEmbedder):
    spec = AdapterSpec(pooling="mean", normalize=True, query_prefix="", document_prefix="", recommended_max_length=2048,
                       requires_remote_code=True)

    def load_tokenizer(self):
        from transformers import AutoTokenizer

        # The tokenizer is part of the reviewed, hash-pinned remote code; the card requires EOS on every sequence.
        return AutoTokenizer.from_pretrained(str(self.snapshot), trust_remote_code=True, add_eos_token=True)


COMPETITOR_ADAPTERS = {
    "jina-v2-code": JinaV2CodeAdapter,
    "codesage": CodeSageAdapter,
    "granite-r2": GraniteR2Adapter,
    "modernbert-embed": ModernBertEmbedAdapter,
    "bge-base": BgeBaseAdapter,
    "qwen3-embed": Qwen3EmbeddingAdapter,
}
