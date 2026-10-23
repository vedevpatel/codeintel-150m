import json

import numpy as np

from codeintel.config import EvaluationConfig, TaskConfig
from codeintel.evaluation.runner import run_evaluation
from codeintel.indexing.storage import build_index
from codeintel.retrieval.bm25 import BM25Index, code_tokens
from codeintel.retrieval.dense import DenseIndex, top_k
from codeintel.retrieval.hybrid import Retriever, aggregate, file_key, reciprocal_rank_fusion, symbol_key

from .conftest import FIXTURES, SAMPLE_REPO, HashingModel


def test_code_tokens_split_identifiers():
    toks = code_tokens("fetchWithRetry(parse_json_file) HTTPServer")
    assert {"fetchwithretry", "fetch", "retry", "parse_json_file", "parse", "json", "file", "httpserver", "http", "server"} <= set(toks)
    assert "with" not in toks  # stopword


def test_bm25_ranks_matching_document_first():
    docs = ["def parse_config(path): return yaml.load(path)", "def send_email(to): smtp.send(to)", "class Cache: pass"]
    bm25 = BM25Index(docs)
    assert bm25.search("parse the yaml config", 3)[0][0] == 0
    assert bm25.search("nothing matches zzz", 3) == []


def test_top_k_is_sorted_with_stable_ties():
    scores = np.array([[0.1, 0.9, 0.5, 0.9]], dtype=np.float32)
    idx, sc = top_k(scores, 3)
    assert idx.tolist() == [[1, 3, 2]]
    assert sc.tolist() == [[np.float32(0.9), np.float32(0.9), np.float32(0.5)]]


def test_dense_index_search():
    emb = np.eye(4, dtype=np.float32)
    idx, _ = DenseIndex(emb, query_block=1).search(np.array([[0, 0, 1, 0], [1, 0, 0, 0]], dtype=np.float32), 2)
    assert idx[:, 0].tolist() == [2, 0]


def test_reciprocal_rank_fusion_rewards_agreement():
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 1, 4]])
    assert [d for d, _ in fused][:2] == [1, 3]


def test_aggregate_keeps_best_rank_per_key():
    index = build_index(SAMPLE_REPO)
    chunks = index.chunks
    paths = [c.path for c in chunks]
    a, b = paths.index("inventory/models.py"), paths.index("inventory/pricing.py")
    hits = [(b, 0.9), (a, 0.8), (b, 0.7)]
    files = aggregate(hits, chunks, file_key)
    assert [h.key for h in files] == ["inventory/pricing.py", "inventory/models.py"]
    assert aggregate(hits, chunks, symbol_key)[0].key.startswith("inventory/pricing.py::")


def test_retriever_modes_on_sample_repo():
    model = HashingModel()
    retriever = Retriever(build_index(SAMPLE_REPO, model=model), model)
    for mode in ("dense", "bm25", "hybrid"):
        chunk, _ = retriever.search("apply a percentage discount to a price", k=3, mode=mode)[0]
        assert chunk.path == "inventory/pricing.py", mode


def test_run_evaluation_on_fixture_cases(tmp_path):
    cfg = EvaluationConfig(
        models=["hashing"],
        tasks=[TaskConfig(name="fixture", type="localization", path=str(FIXTURES / "retrieval_cases.jsonl"), repos_dir=str(FIXTURES))],
        output_dir=str(tmp_path),
        bootstrap_samples=100,
        cache_dir=str(tmp_path / "cache"),
    )
    results = run_evaluation(cfg, model_loader=lambda _: HashingModel())
    levels = results["tasks"]["fixture"]["levels"]
    assert set(levels["file"]["systems"]) == {"bm25", "hashing/dense", "hashing/hybrid"}
    assert levels["file"]["systems"]["bm25"]["metrics"]["hit@5"] >= 0.75
    assert levels["symbol"]["systems"]["bm25"]["metrics"]["num_queries"] == 12
    assert (tmp_path / "report.md").read_text().startswith("# Evaluation report")
    rows = [json.loads(line) for line in open(tmp_path / "per_query.jsonl")]
    assert {r["system"] for r in rows} == {"bm25", "hashing/dense", "hashing/hybrid"}


def test_run_evaluation_retrieval_task(tmp_path):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    docs = {"d1": "def add(a, b): return a + b", "d2": "def read_json(path): return json.load(open(path))", "d3": "class Stack: pass"}
    (task_dir / "corpus.jsonl").write_text("".join(json.dumps({"id": k, "text": v}) + "\n" for k, v in docs.items()))
    (task_dir / "queries.jsonl").write_text(json.dumps({"id": "q1", "text": "load json from a path"}) + "\n")
    (task_dir / "qrels.jsonl").write_text(json.dumps({"query_id": "q1", "doc_id": "d2"}) + "\n")
    cfg = EvaluationConfig(
        models=["hashing"], tasks=[TaskConfig(name="toy", type="retrieval", path=str(task_dir))], output_dir=str(tmp_path / "out"),
        bootstrap_samples=50, cache_dir=str(tmp_path / "cache"),
    )
    results = run_evaluation(cfg, model_loader=lambda _: HashingModel())
    systems = results["tasks"]["toy"]["levels"]["doc"]["systems"]
    assert systems["bm25"]["metrics"]["hit@1"] == 1.0
    assert systems["hashing/dense"]["metrics"]["mrr"] == 1.0


def test_top_k_ties_at_cutoff_prefer_lowest_index():
    scores = np.array([[0.5, 0.9, 0.5, 0.5]], dtype=np.float32)
    assert top_k(scores, 2)[0].tolist() == [[1, 0]]
