"""Command-line interface: `codeintel {index,search,evaluate,validate,train,export}`."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def _load_model(path: str | None):
    if not path:
        return None
    from codeintel.config import load_model_config
    from codeintel.models.base import load_model

    return load_model(load_model_config(path))


def cmd_index(args: argparse.Namespace) -> int:
    from codeintel.indexing.storage import EmbeddingCache, build_index

    model = _load_model(args.model)
    cache = EmbeddingCache(args.cache_dir, model.fingerprint, model.dim) if model and args.cache_dir else None
    index = build_index(args.repo, model=model, cache=cache)
    index.save(args.out)
    print(f"indexed {len(index.manifest['files'])} files into {len(index.chunks)} chunks -> {args.out}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    from codeintel.indexing.storage import CodeIndex
    from codeintel.retrieval.hybrid import Retriever, aggregate, file_key, symbol_key

    index = CodeIndex.load(args.index)
    model = _load_model(args.model) if args.mode != "bm25" else None
    if model is not None and index.manifest.get("model_fingerprint") != model.fingerprint:
        print(
            f"error: index was built with {index.manifest.get('model_fingerprint')!r}, "
            f"not {model.fingerprint!r}; re-index or pass the matching --model",
            file=sys.stderr,
        )
        return 2
    retriever = Retriever(index, model)
    depth = args.k if args.level == "chunk" else max(200, args.k)
    hits = retriever.search_many([args.query], k=depth, mode=args.mode)[0]
    if args.level == "chunk":
        for rank, (i, score) in enumerate(hits, 1):
            c = index.chunks[i]
            print(f"{rank:3d}  {score:.4f}  {c.path}:{c.start_line}-{c.end_line}  {c.symbol or '<module>'}")
    else:
        key = file_key if args.level == "file" else symbol_key
        for h in aggregate(hits, index.chunks, key, k=args.k):
            print(f"{h.rank:3d}  {h.score:.4f}  {h.key}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from codeintel.config import load_evaluation_config
    from codeintel.evaluation.runner import run_evaluation

    cfg = load_evaluation_config(args.config)
    if args.models:
        cfg.models = args.models
    results = run_evaluation(cfg, output_dir=args.output_dir)
    out = Path(args.output_dir or cfg.output_dir)
    print((out / "report.md").read_text())
    return 0 if results["tasks"] else 1


def cmd_validate(args: argparse.Namespace) -> int:
    from codeintel.data import validation

    if args.kind == "training":
        report = validation.validate_training(args.path)
    elif args.kind == "localization":
        report = validation.validate_localization(args.path, args.repos_dir)
    else:
        report = validation.validate_retrieval_dir(args.path)
    print(report.summary())
    for e in report.errors[:50]:
        print(f"  error: {e}")
    for w in report.warnings[:20]:
        print(f"  warning: {w}")
    return 0 if report.ok else 1


def cmd_train(args: argparse.Namespace) -> int:
    from codeintel.config import load_training_config
    from codeintel.training.trainer import train

    cfg = load_training_config(args.config)
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.output_dir:
        cfg.output_dir = args.output_dir
    final = train(cfg)
    print(f"saved {final}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from codeintel.config import load_model_config
    from codeintel.models.base import load_model
    from codeintel.models.export import export_onnx

    cfg = load_model_config(args.model)
    cfg.device = "cpu"
    report = export_onnx(
        load_model(cfg), args.out, opset=args.opset, quantize=not args.no_quantize, full_int8=args.full_int8
    )
    print(json.dumps(report, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="codeintel", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("index", help="chunk (and embed) a repository")
    p.add_argument("repo")
    p.add_argument("--out", required=True)
    p.add_argument("--model", help="model YAML; omit for a lexical-only index")
    p.add_argument("--cache-dir", default="artifacts/cache")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("search", help="query an index")
    p.add_argument("index")
    p.add_argument("query")
    p.add_argument("--model")
    p.add_argument("--mode", choices=["dense", "bm25", "hybrid"], default="hybrid")
    p.add_argument("--level", choices=["chunk", "file", "symbol"], default="chunk")
    p.add_argument("-k", type=int, default=10)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("evaluate", help="run an evaluation config")
    p.add_argument("config")
    p.add_argument("--models", nargs="+", help="override the config's model list")
    p.add_argument("--output-dir")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("validate", help="validate a dataset file or directory")
    p.add_argument("kind", choices=["training", "localization", "retrieval"])
    p.add_argument("path")
    p.add_argument("--repos-dir")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("train", help="contrastive fine-tuning")
    p.add_argument("config")
    p.add_argument("--max-steps", type=int)
    p.add_argument("--output-dir")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("export", help="export a model to ONNX (+ INT8)")
    p.add_argument("model")
    p.add_argument("--out", required=True)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--no-quantize", action="store_true")
    p.add_argument("--full-int8", action="store_true", help="also quantize layers the adapter keeps in fp32")
    p.set_defaults(func=cmd_export)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
