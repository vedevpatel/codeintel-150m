"""Export a model config to ONNX (fp32 + mixed-precision INT8) and verify against PyTorch.

  uv run python scripts/export_onnx.py configs/models/gte-modernbert.yaml --out artifacts/onnx/gte-modernbert

The INT8 graph quantizes MatMul weights per channel and keeps the adapter's outlier-heavy layers
(`AdapterSpec.int8_fp32_nodes`, the FFN down-projections) in fp32; pass --full-int8 to quantize everything.
Load the result with a model config of `adapter: onnx`, `local_dir: <out>` and `dtype: int8` or `float32`.
"""

from __future__ import annotations

import sys

from codeintel.cli import main

if __name__ == "__main__":
    sys.exit(main(["export", *sys.argv[1:]]))
