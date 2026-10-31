"""Assemble a Hugging Face model repository folder from an exported checkpoint (+ optional ONNX export).

  uv run python scripts/build_hf_release.py --model artifacts/run/granite-r2/model/granite-r2-ft \
      --onnx artifacts/onnx/<model> --out artifacts/hf/codeintel

Writes the transformers files, the sentence-transformers module files (CLS pooling + L2 normalisation, as the adapter
spec says), and onnx/model.onnx + onnx/model_int8.onnx. The model card (README.md) is written separately.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

CODE = [
    "# src/flask/app.py::Flask.run\ndef run(self, host=None, port=None, debug=None):\n    ...",
    "# src/utils/date.ts::parseDate\nexport function parseDate(s: string): Date { return new Date(s); }",
]
QUERIES = ["Server ignores the port argument when debug is enabled", "dates parsed in the wrong timezone"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="exported HF checkpoint directory (with codeintel_adapter.json)")
    ap.add_argument("--onnx", help="ONNX export directory (model.onnx, model_int8.onnx)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seq-length", type=int, default=1024)
    args = ap.parse_args()
    src, out = Path(args.model), Path(args.out)
    spec = json.loads((src / "codeintel_adapter.json").read_text())
    assert spec["pooling"] == "cls" and spec["normalize"], spec
    out.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"):
        shutil.copy(src / name, out / name)
    for name in ("special_tokens_map.json",):
        if (src / name).exists():
            shutil.copy(src / name, out / name)
    (out / "modules.json").write_text(json.dumps([
        {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
        {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
        {"idx": 2, "name": "2", "path": "2_Normalize", "type": "sentence_transformers.models.Normalize"},
    ], indent=2) + "\n")
    (out / "sentence_bert_config.json").write_text(json.dumps({"max_seq_length": args.max_seq_length, "do_lower_case": False}, indent=2) + "\n")
    (out / "1_Pooling").mkdir(exist_ok=True)
    hidden = json.loads((src / "config.json").read_text())["hidden_size"]
    (out / "1_Pooling/config.json").write_text(json.dumps({
        "word_embedding_dimension": hidden, "pooling_mode_cls_token": True, "pooling_mode_mean_tokens": False,
        "pooling_mode_max_tokens": False, "pooling_mode_mean_sqrt_len_tokens": False, "include_prompt": True}, indent=2) + "\n")
    (out / "2_Normalize").mkdir(exist_ok=True)
    (out / "config_sentence_transformers.json").write_text(json.dumps({
        "prompts": {"query": spec["query_prefix"], "document": spec["document_prefix"]}, "default_prompt_name": None,
        "similarity_fn_name": "cosine"}, indent=2) + "\n")
    if args.onnx:
        (out / "onnx").mkdir(exist_ok=True)
        for name in ("model.onnx", "model_int8.onnx"):
            shutil.copy(Path(args.onnx) / name, out / "onnx" / name)
    print("wrote", sorted(p.name for p in out.iterdir()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
