import json

from codeintel.data.schemas import TrainingPair
from codeintel.data.splits import assign_split, group_leakage, split_by_group, temporal_split
from codeintel.data.validation import validate_localization, validate_retrieval_dir, validate_training
from codeintel.evaluation.contamination import decontaminate, exact_key, overlap_report


def test_assign_split_is_deterministic_and_roughly_proportional():
    names = [assign_split(f"repo{i}") for i in range(4000)]
    assert names == [assign_split(f"repo{i}") for i in range(4000)]
    assert 0.87 < names.count("train") / len(names) < 0.93


def test_split_by_group_has_no_leakage_and_honours_pins():
    records = [{"repo": f"r{i % 50}", "i": i} for i in range(500)]
    splits = split_by_group(records, lambda r: r["repo"], pinned={"r1": "test"})
    assert group_leakage(splits, lambda r: r["repo"]) == {}
    assert all(r["repo"] != "r1" for r in splits["train"])


def test_temporal_split():
    before, after = temporal_split([{"d": "2024-05-01"}, {"d": "2025-06-01"}, {"d": ""}], lambda r: r["d"], "2025-01-01")
    assert len(before) == 2 and len(after) == 1


def test_exact_key_ignores_comments_and_whitespace():
    assert exact_key("def f(x):  # add\n    return x+1") == exact_key("def f(x):\n  return x+1  // c")


def test_overlap_report_finds_exact_and_near_duplicates():
    base = "def compute_total(items):\n    total = 0\n    for item in items:\n        total += item.price * item.qty\n    return total\n"
    near = base.replace("total += item.price * item.qty", "total += item.price * item.quantity")
    other = "class Router:\n    def route(self, path):\n        return self.table.get(path)\n"
    report = overlap_report([("e1", base), ("e2", near), ("e3", other)], [("t1", base + "\n# trailing comment")], threshold=0.5)
    assert report.exact == 1 and report.near_duplicate == 1


def test_decontaminate_drops_overlapping_records():
    recs = [
        TrainingPair(query="sum the order", positive="def total(o): return sum(o)", task="nl2code", source="s", repo="a"),
        TrainingPair(query="route a path", positive="def route(p): return table[p]", task="nl2code", source="s", repo="a"),
        TrainingPair(query="anything", positive="def x(): pass", task="nl2code", source="s", repo="blocked"),
    ]
    kept, stats = decontaminate(recs, ["def total(o):\n    return sum(o)"], blocked_repos={"blocked"})
    assert [r.query for r in kept] == ["route a path"]
    assert stats == {"exact": 1, "near_duplicate": 0, "repo": 1, "kept": 1}


def test_validate_training(tmp_path):
    path = tmp_path / "t.jsonl"
    rows = [
        {"query": "parse json", "positive": "def parse(): ...", "task": "nl2code", "source": "docstring"},
        {"query": "same", "positive": "same", "task": "nl2code", "source": "docstring"},
        {"query": "q", "positive": "p", "task": "nonsense", "source": "x"},
        {"query": "q", "positive": "p", "task": "nl2code", "source": "x", "typo_field": 1},
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    report = validate_training(path)
    assert not report.ok
    assert any("trivial" in e for e in report.errors)
    assert any("unknown task" in e for e in report.errors)
    assert any("unknown fields" in e for e in report.errors)


def test_validate_localization_paths(tmp_path):
    path = tmp_path / "c.jsonl"
    rows = [
        {"id": "a", "repo": "r", "query": "bug", "gold_files": ["../etc/passwd"]},
        {"id": "a", "repo": "r", "query": "bug", "gold_files": ["x.py"], "gold_symbols": ["y.py::f"], "commit": "nothex"},
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    report = validate_localization(path)
    joined = "\n".join(report.errors)
    assert "relative POSIX path" in joined and "duplicate case id" in joined
    assert "gold symbol" in joined and "not a hex SHA" in joined


def test_validate_retrieval_dir(tmp_path):
    (tmp_path / "corpus.jsonl").write_text(json.dumps({"id": "d1", "text": "x"}) + "\n")
    (tmp_path / "queries.jsonl").write_text(json.dumps({"id": "q1", "text": "y"}) + "\n" + json.dumps({"id": "q2", "text": "z"}) + "\n")
    (tmp_path / "qrels.jsonl").write_text(json.dumps({"query_id": "q1", "doc_id": "d9"}) + "\n")
    report = validate_retrieval_dir(tmp_path)
    assert any("unknown document" in e for e in report.errors)
    assert any("no positive judgement" in w for w in report.warnings)


def test_strip_python_docstring_for_codesearchnet():
    from codeintel.data.public import strip_python_docstring

    method = 'def f(self, a):\n        """\n        Doc here.\n        """\n        return a\n'
    assert strip_python_docstring(method) == "def f(self, a):\n        return a\n"
    py2 = 'def f(a):\n    """Doc."""\n    print "py2"\n'
    assert "Doc" not in strip_python_docstring(py2) and 'print "py2"' in strip_python_docstring(py2)
    assert strip_python_docstring("def f(a): return a\n") == "def f(a): return a\n"


def test_collapse_duplicates_remaps_judgements():
    from codeintel.data.public import collapse_duplicates
    from codeintel.data.schemas import CorpusDoc, Qrel

    corpus = [CorpusDoc("d1", "x"), CorpusDoc("d2", "y"), CorpusDoc("d3", "x")]
    kept, qrels, merged = collapse_duplicates(corpus, [Qrel("q1", "d3"), Qrel("q2", "d2")])
    assert [d.id for d in kept] == ["d1", "d2"] and merged == 1
    assert sorted((q.query_id, q.doc_id) for q in qrels) == [("q1", "d1"), ("q2", "d2")]
