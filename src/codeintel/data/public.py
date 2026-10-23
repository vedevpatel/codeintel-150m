"""Import public code-retrieval benchmarks into the local retrieval-task format.

Only Parquet data files are downloaded (no dataset loading scripts are executed), at pinned dataset commits.
These sets are for evaluation; their licences derive from the underlying GitHub code and are not uniform,
so do not redistribute or train on them.
"""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from pathlib import Path

from codeintel.data.schemas import CorpusDoc, Qrel, QueryRecord, write_jsonl

COIR_REVISIONS = {
    "CoIR-Retrieval/CodeSearchNet": "25e0292562b7bee26dd9b2d83a03981795862c77",
    "CoIR-Retrieval/cosqa": "0846fa3b963a21bead36e9fab61451fc83b777a6",
}


def _read_parquet(repo_id: str, filename: str) -> list[dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id, filename, repo_type="dataset", revision=COIR_REVISIONS[repo_id])
    return pq.read_table(path).to_pylist()


_PY_DOCSTRING = re.compile(r"""^([ \t]*)[rRuUbB]{0,2}(\"\"\"|\'\'\')(.*?)\2[ \t]*\n?""", re.S | re.M)


def strip_python_docstring(code: str) -> str:
    """Remove the leading docstring of a single function/class snippet (CSN code embeds the query in it)."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        tree = None
    if tree is not None:
        node = next((n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))), None)
        if node is not None and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                lines = code.splitlines(keepends=True)
                if first.lineno > node.lineno:  # docstring on its own lines
                    return "".join(lines[: first.lineno - 1] + lines[first.end_lineno :])
        return code
    # Python 2 or otherwise unparsable: drop the first triple-quoted string after the signature.
    header_end = code.find(":\n")
    if header_end < 0:
        return code
    head, body = code[: header_end + 2], code[header_end + 2 :]
    return head + _PY_DOCSTRING.sub("", body, count=1)


def collapse_duplicates(corpus: list[CorpusDoc], qrels: list[Qrel]) -> tuple[list[CorpusDoc], list[Qrel], int]:
    """Merge documents with identical text into the first id, remapping judgements.

    Identical copies receive identical scores, so their order (and whether the judged copy ranks first) is
    decided by tie-breaking rather than by the model. CoIR's CoSQA corpus is 70% exact duplicates.
    """
    canonical: dict[str, str] = {}
    remap: dict[str, str] = {}
    kept: list[CorpusDoc] = []
    for doc in corpus:
        first = canonical.setdefault(doc.text, doc.id)
        remap[doc.id] = first
        if first == doc.id:
            kept.append(doc)
    merged: dict[tuple[str, str], float] = {}
    for q in qrels:
        key = (q.query_id, remap.get(q.doc_id, q.doc_id))
        merged[key] = max(merged.get(key, 0.0), q.relevance)
    return kept, [Qrel(qid, did, rel) for (qid, did), rel in merged.items()], len(corpus) - len(kept)


def import_codesearchnet(language: str, out_dir: str | Path, split: str = "test") -> dict[str, int]:
    """CodeSearchNet NL -> code for one language.

    CoIR publishes CSN in the code -> docstring direction (queries are code, corpus is docstrings). The
    standard NL -> code task is obtained by inverting it: docstrings become queries and the code of the same
    split becomes the corpus. Python snippets embed their docstring (i.e. the query) in the code, so it is
    stripped; JavaScript snippets start at the function and need no change.
    """
    repo = "CoIR-Retrieval/CodeSearchNet"
    qrels = _read_parquet(repo, f"{language}-qrels/{split}-00000-of-00001.parquet")
    code_ids = {r["query-id"] for r in qrels}
    doc_ids = {r["corpus-id"] for r in qrels}
    code = {r["_id"]: r for r in _read_parquet(repo, f"{language}-queries/queries-00000-of-00001.parquet") if r["_id"] in code_ids}
    docs = {r["_id"]: r for r in _read_parquet(repo, f"{language}-corpus/corpus-00000-of-00001.parquet") if r["_id"] in doc_ids}
    out_dir = Path(out_dir)
    strip = strip_python_docstring if language == "python" else (lambda c: c)
    corpus = [CorpusDoc(id=cid, text=strip(code[cid]["text"]), language=language) for cid in sorted(code)]
    by_doc: dict[str, list[str]] = defaultdict(list)
    for r in qrels:
        if r["query-id"] in code and r["corpus-id"] in docs:
            by_doc[r["corpus-id"]].append(r["query-id"])
    queries = [QueryRecord(id=did, text=docs[did]["text"], language=language) for did in sorted(by_doc)]
    judgements = [Qrel(query_id=did, doc_id=cid) for did, cids in sorted(by_doc.items()) for cid in cids]
    corpus, judgements, merged = collapse_duplicates(corpus, judgements)
    return {
        "merged_duplicates": merged,
        "corpus": write_jsonl(out_dir / "corpus.jsonl", corpus),
        "queries": write_jsonl(out_dir / "queries.jsonl", queries),
        "qrels": write_jsonl(out_dir / "qrels.jsonl", judgements),
    }


def import_cosqa(out_dir: str | Path, split: str = "test") -> dict[str, int]:
    """CoSQA: real web search queries -> Python functions (NL -> code)."""
    repo = "CoIR-Retrieval/cosqa"
    qrels = [r for r in _read_parquet(repo, f"data/{split}-00000-of-00001.parquet") if float(r["score"]) > 0]
    qids = {r["query-id"] for r in qrels}
    queries = [
        QueryRecord(id=r["_id"], text=r["text"], language="python")
        for r in _read_parquet(repo, "queries/queries-00000-of-00001.parquet")
        if r["_id"] in qids
    ]
    corpus = [
        CorpusDoc(id=r["_id"], text=r["text"], language="python")
        for r in _read_parquet(repo, "corpus/corpus-00000-of-00001.parquet")
    ]
    judgements = [Qrel(r["query-id"], r["corpus-id"], float(r["score"])) for r in qrels]
    corpus, judgements, merged = collapse_duplicates(corpus, judgements)
    out_dir = Path(out_dir)
    return {
        "merged_duplicates": merged,
        "corpus": write_jsonl(out_dir / "corpus.jsonl", corpus),
        "queries": write_jsonl(out_dir / "queries.jsonl", queries),
        "qrels": write_jsonl(out_dir / "qrels.jsonl", judgements),
    }
