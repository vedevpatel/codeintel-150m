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


def test_localization_breakdown_by_tags(tmp_path):
    cases = [json.loads(line) for line in open(FIXTURES / "retrieval_cases.jsonl")]
    for i, c in enumerate(cases):
        c["tags"] = {"language": c["language"], "mentions": "lexical" if i % 2 else "descriptive"}
    path = tmp_path / "cases.jsonl"
    path.write_text("".join(json.dumps(c) + "\n" for c in cases))
    cfg = EvaluationConfig(
        models=["hashing"], tasks=[TaskConfig(name="t", type="localization", path=str(path), repos_dir=str(FIXTURES))],
        output_dir=str(tmp_path / "out"), bootstrap_samples=50, cache_dir=str(tmp_path / "cache"),
    )
    results = run_evaluation(cfg, model_loader=lambda _: HashingModel())
    levels = results["tasks"]["t"]["levels"]
    assert {"symbol[mentions=descriptive]", "symbol[language=python,mentions=lexical]", "file[language=typescript]"} <= set(levels)
    n = levels["symbol[mentions=descriptive]"]["systems"]["bm25"]["metrics"]["num_queries"]
    assert n == 6
    report = (tmp_path / "out" / "report.md").read_text()
    assert report.index("### symbol-level") < report.index("### file-level")
    assert "Breakdown (symbol-level" in report


def test_localization_resumes_from_per_repo_checkpoints(tmp_path):
    cfg = EvaluationConfig(
        models=["hashing"],
        tasks=[TaskConfig(name="fixture", type="localization", path=str(FIXTURES / "retrieval_cases.jsonl"), repos_dir=str(FIXTURES))],
        output_dir=str(tmp_path), bootstrap_samples=50, cache_dir=str(tmp_path / "cache"),
    )
    first = run_evaluation(cfg, model_loader=lambda _: HashingModel())
    assert (tmp_path / "checkpoints" / "fixture" / "sample_repo.json").exists()
    second = run_evaluation(cfg, model_loader=lambda _: HashingModel())
    assert second["tasks"]["fixture"]["stats"]["repos_resumed_from_checkpoint"] == 1
    a = first["tasks"]["fixture"]["levels"]["symbol"]["systems"]
    b = second["tasks"]["fixture"]["levels"]["symbol"]["systems"]
    assert {k: v["metrics"] for k, v in a.items()} == {k: v["metrics"] for k, v in b.items()}


def test_checkpoint_transport_roundtrip_and_damage_detection():
    from codeintel.evaluation.transport import pack, parse_stream

    part = {
        "key": "k1", "file_qrels": {"q1": {"a.py": 1.0}}, "symbol_qrels": {"q1": {"a.py::f": 1.0}},
        "file_rank": {"bm25": {"q1": ["a.py", "b.py"]}, "m/dense": {"q1": ["b.py", "a.py"]}},
        "sym_rank": {"bm25": {"q1": ["a.py::f", "b.py::g"]}, "m/dense": {"q1": ["b.py::g"]}},
        "missing_gold": 0, "skipped": 0, "chunks": 10, "snapshots": 1,
    }
    block = pack("repoA", part) + pack("repoB", {**part, "key": "k2"})
    noisy = "INFO starting\n" + block.replace("@@CHECKPOINT repoB", "progress 50%\n@@CHECKPOINT repoB") + "done\n"
    found, problems = parse_stream(noisy)
    assert not problems and found["repoA"] == part and found["repoB"]["key"] == "k2"
    # A cut-off job still yields the finished repository and reports the damage.
    cut = block[: block.index("@@CHECKPOINT repoB") + 200]
    found, problems = parse_stream(cut)
    assert list(found) == ["repoA"] and problems
    # Flipped payload characters fail the CRC.
    tampered = block.replace("@@END\n", "@@END\n", 1)
    i = tampered.index("\n", tampered.index("@@CHECKPOINT")) + 5
    tampered = tampered[:i] + ("A" if tampered[i] != "A" else "B") + tampered[i + 1 :]
    assert "repoA: block corrupted or truncated" in parse_stream(tampered)[1]


def test_checkpoints_are_portable_across_environments(tmp_path):
    """Rankings keyed by model identity resume under a different fingerprint (library versions, dtype)."""
    class Versioned(HashingModel):
        def __init__(self, fingerprint):
            super().__init__()
            self._fp = fingerprint

        @property
        def fingerprint(self):
            return self._fp

        identity = "hashing-identity"

    def run(fp, out):
        cfg = EvaluationConfig(
            models=["m"], tasks=[TaskConfig(name="t", type="localization", path=str(FIXTURES / "retrieval_cases.jsonl"), repos_dir=str(FIXTURES))],
            output_dir=str(out), bootstrap_samples=20, cache_dir=str(tmp_path / f"cache-{fp}"),
        )
        seen = []
        res = run_evaluation(cfg, model_loader=lambda _: Versioned(fp), on_repo_done=lambda repo, ck: seen.append((repo, ck["key"])))
        return res, seen

    first, seen1 = run("torch2.14|bf16", tmp_path / "a")
    import shutil

    shutil.copytree(tmp_path / "a" / "checkpoints", tmp_path / "b" / "checkpoints")
    second, seen2 = run("torch2.3|fp16", tmp_path / "b")
    assert second["tasks"]["t"]["stats"]["repos_resumed_from_checkpoint"] == 1
    assert seen1 == seen2 and seen1[0][0] == "sample_repo"


def test_bm25_array_build_matches_counter_build():
    from collections import Counter

    from codeintel.retrieval.bm25 import BM25Index, code_tokens, term_arrays

    docs = ["def parse_config(path): return yaml.load(path)", "def send_email(to): smtp.send(to)", "class Cache: pass", "", "parse parse parse yaml"]
    counter_built = BM25Index(docs)
    vocab: dict[str, int] = {}
    arrays = [term_arrays(Counter(code_tokens(d)), vocab) for d in docs]
    array_built = BM25Index(vocab=vocab, arrays=arrays)
    for q in ["parse yaml config", "send email", "cache", "nothing zzz", ""]:
        np.testing.assert_allclose(counter_built.scores(q), array_built.scores(q), rtol=1e-6)
    # A vocabulary that grew after the index was built (a later snapshot) must not break lookups.
    vocab["brand_new_term"] = len(vocab)
    assert array_built.scores("brand_new_term").sum() == 0


def test_vector_memo_skips_cache_and_encoder_for_known_texts(hashing_model):
    from codeintel.indexing.storage import encode_documents_cached

    memo: dict = {}
    first = encode_documents_cached(hashing_model, ["a b", "c d", "a b"], memo=memo)
    assert hashing_model.encoded == 2 and set(memo) == {"a b", "c d"}
    second = encode_documents_cached(hashing_model, ["c d", "e f", "a b"], memo=memo)
    assert hashing_model.encoded == 3  # only "e f" was new
    np.testing.assert_allclose(second[0], first[1])
    np.testing.assert_allclose(second[2], first[0])


def test_shard_assignment_balances_load_and_covers_every_repo():
    from codeintel.evaluation.shard import assign_shards

    costs = {"big": 100, "a": 40, "b": 35, "c": 30, "d": 5}
    shards = assign_shards(costs, 3)
    assert sorted(r for s in shards for r in s) == sorted(costs)
    loads = sorted(sum(costs[r] for r in s) for s in shards)
    assert loads[-1] == 100  # the biggest repository alone; the rest are spread over the other workers
    assert assign_shards({"only": 1}, 4) == [["only"]]  # empty shards are dropped


def test_block_relay_keeps_blocks_whole_across_threads():
    import io
    import threading

    from codeintel.evaluation.shard import BlockRelay
    from codeintel.evaluation.transport import END, pack, parse_stream

    part = {"key": "k", "file_qrels": {}, "symbol_qrels": {}, "file_rank": {"bm25": {"q": ["a", "b"]}},
            "sym_rank": {"bm25": {"q": ["a::f"]}}, "missing_gold": 0, "skipped": 0, "chunks": 1, "snapshots": 1}

    def worker_stream(repo, n):
        text = ""
        for i in range(n):
            text += f"log line {i}\n" + pack(f"{repo}{i}", part)
        return io.StringIO(text)

    out = io.StringIO()
    relay = BlockRelay(out)
    threads = [threading.Thread(target=relay.pump, args=(f"w{i}", worker_stream(f"r{i}_", 20))) for i in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    found, problems = parse_stream(out.getvalue())
    assert not problems and len(found) == 80 and not relay.truncated
    # A worker that dies inside a block must not poison the stream.
    cut = io.StringIO(pack("r", part)[:-len(END) - 5])
    relay2 = BlockRelay(io.StringIO())
    relay2.pump("w9", cut)
    assert relay2.truncated == ["w9"]


def test_weighted_hybrid_modes_change_the_fusion_and_the_report_pairs_them():
    from codeintel.retrieval.hybrid import split_mode

    assert split_mode("hybrid") == ("hybrid", None) and split_mode("hybrid@2.5") == ("hybrid", 2.5)
    model = HashingModel()
    retriever = Retriever(build_index(SAMPLE_REPO, model=model), model)
    q = "discount percent for a price"
    even = retriever.search_many([q], k=15, mode="hybrid")[0]
    dense_heavy = retriever.search_many([q], k=15, mode="hybrid@50")[0]
    bm25_heavy = retriever.search_many([q], k=15, mode="hybrid@0.02")[0]
    dense_only = retriever.search_many([q], k=15, mode="dense")[0]
    bm25_only = retriever.search_many([q], k=15, mode="bm25")[0]
    assert [i for i, _ in dense_heavy[:5]] == [i for i, _ in dense_only[:5]]
    assert [i for i, _ in bm25_heavy[:5]] == [i for i, _ in bm25_only[:5]]
    assert len({tuple(i for i, _ in r) for r in (even, dense_heavy, bm25_heavy)}) >= 2


def test_run_evaluation_reports_weighted_hybrids_and_pairs_them(tmp_path):
    cfg = EvaluationConfig(
        models=["a", "b"], tasks=[TaskConfig(name="t", type="localization", path=str(FIXTURES / "retrieval_cases.jsonl"), repos_dir=str(FIXTURES))],
        output_dir=str(tmp_path), bootstrap_samples=20, cache_dir=str(tmp_path / "cache"),
        retrieval_modes=["dense", "bm25", "hybrid@0.5", "hybrid@2"],
    )
    names = iter(["m1", "m2"])

    def loader(_):
        m = HashingModel()
        m.name = next(names)
        return m

    res = run_evaluation(cfg, model_loader=loader)
    level = res["tasks"]["t"]["levels"]["symbol"]
    assert {"bm25", "m1/dense", "m1/hybrid@0.5", "m1/hybrid@2", "m2/hybrid@2"} <= set(level["systems"])
    assert any(c["a"] == "m1/hybrid@2" and c["b"] == "m2/hybrid@2" for c in level["comparisons"])
