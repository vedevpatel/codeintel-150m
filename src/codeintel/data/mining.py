"""Mine training pairs and localization cases from git repositories.

Pair types (TrainingPair.task / .source):
  nl2code    / docstring   docstring or JSDoc summary -> the definition with its doc comment removed
  code2code  / tests       Python test function -> the implementation it is named after
  navigation / calls       a call site (a few lines of context) -> the called definition
  issue2code / commits     commit message -> code changed by the commit, as it was before the change

Positives are rendered exactly like index chunks (same header and dedent), so training matches inference.
"""

from __future__ import annotations

import random
import re
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from codeintel.data.schemas import LocalizationCase, TrainingPair
from codeintel.indexing.chunker import Chunker
from codeintel.indexing.parser import parse_symbols
from codeintel.indexing.repository import (
    DEFAULT_LANGUAGES,
    _skipped_dir,
    detect_language,
    is_probably_generated,
    iter_source_files_at_commit,
)
from codeintel.schemas import Chunk, SourceFile, Symbol, split_lines

TEST_PATH = re.compile(
    r"(^|/)(tests?|__tests__|testing|spec)/|(^|/)test_[^/]*\.py$|_test\.py$|\.(test|spec)\.[jt]sx?$|(^|/)conftest\.py$"
)
DOC_FIELD = re.compile(r"^\s*(:param|:return|:raises|:type|@param|@returns?|@throws|Args:|Returns:|Raises:|Example)", re.I)
TRIVIAL_SUBJECT = re.compile(
    r"^(merge|revert|bump|release|prepare release|v?\d+\.\d+|update (changelog|readme|docs?|deps|dependencies)|"
    r"chore\((deps|release)\)|chore: (release|bump)|\[pre-commit|pre-commit autoupdate|wip\b|typo|format|lint|"
    r"version packages)",
    re.I,
)
TRAILER = re.compile(r"^(signed-off-by|co-authored-by|reviewed-by|acked-by|change-id|cc|ghstack-source-id|pull request resolved):", re.I)
PR_REF = re.compile(r"\s*\(#\d+\)")


def is_test_path(path: str) -> bool:
    return bool(TEST_PATH.search(path))


def doc_summary(doc: str, max_chars: int = 600) -> str:
    """First paragraph of a docstring, without parameter/return sections."""
    lines = []
    for line in doc.strip().splitlines():
        if not line.strip():
            if lines:
                break
            continue
        if DOC_FIELD.match(line):
            break
        lines.append(line.strip())
    return " ".join(lines)[:max_chars].strip()


def _strip_lines(lines: list[str], start: int, end: int, drop: tuple[int, int] | None) -> str:
    keep = [
        line
        for n, line in enumerate(lines[start - 1 : end], start)
        if drop is None or not (drop[0] <= n <= drop[1])
    ]
    return "\n".join(keep)


# ------------------------------------------------------------------------------- snapshot mining


@dataclass
class SnapshotPairs:
    docstring: list[TrainingPair]
    tests: list[TrainingPair]
    calls: list[TrainingPair]
    # Non-test functions/methods without a doc comment (distractors for docstring retrieval tasks).
    undocumented: list[Chunk]


def _leaf_symbols(symbols: list[Symbol]) -> list[Symbol]:
    out = []
    for s in symbols:
        contains_other = any(
            o is not s and s.start_line <= o.start_line and o.end_line <= s.end_line for o in symbols
        )
        if not contains_other:
            out.append(s)
    return out


def mine_snapshot(
    files: Iterable[SourceFile],
    repo: str,
    chunker: Chunker | None = None,
    max_call_pairs: int = 2000,
    min_doc_words: int = 4,
    min_body_lines: int = 3,
) -> SnapshotPairs:
    chunker = chunker or Chunker()
    docstring: list[TrainingPair] = []
    definitions: dict[str, list[Chunk]] = defaultdict(list)  # bare name -> definition chunks (non-test)
    test_chunks: list[tuple[Symbol, Chunk]] = []
    impl_chunks: list[tuple[SourceFile, Chunk]] = []
    undocumented: list[Chunk] = []
    for source in files:
        symbols = parse_symbols(source)
        lines = split_lines(source.text)
        leaves = {s.qualified_name: s for s in _leaf_symbols(symbols)}
        testy = is_test_path(source.path)
        for chunk in chunker.chunk(source, symbols):
            sym = leaves.get(chunk.symbol or "")
            if sym is None or chunk.kind not in ("function", "method") or "(part " in chunk.text.split("\n", 1)[0]:
                continue
            if testy:
                test_chunks.append((sym, chunk))
                continue
            definitions[sym.name].append(chunk)
            impl_chunks.append((source, chunk))
            code = _strip_lines(lines, sym.start_line, sym.end_line, sym.doc_span)
            if len([x for x in code.splitlines() if x.strip()]) < min_body_lines:
                continue  # signatures without bodies (interface members, overloads, stubs)
            if sym.doc_span is None:
                undocumented.append(chunk)
                continue
            summary = doc_summary(sym.docstring)
            if len(set(summary.lower().split())) >= min_doc_words:
                positive = chunker.make_chunk(source, sym.start_line, sym.end_line, sym.kind, sym.qualified_name, code)
                docstring.append(
                    TrainingPair(
                        query=summary,
                        positive=positive.text,
                        task="nl2code",
                        source="docstring",
                        language=source.language,
                        repo=repo,
                        id=f"{repo}:{positive.chunk_id}",
                    )
                )

    unique_defs = {name: chunks[0] for name, chunks in definitions.items() if len(chunks) == 1 and len(name) >= 4}

    tests: list[TrainingPair] = []
    for sym, chunk in test_chunks:
        if not sym.name.startswith("test_"):
            continue
        target = sym.name[len("test_") :]
        # Longest defined name that the test name starts with: test_parse_url_empty -> parse_url.
        candidates = [n for n in unique_defs if target == n or target.startswith(n + "_")]
        if not candidates:
            continue
        impl = unique_defs[max(candidates, key=len)]
        tests.append(
            TrainingPair(
                query=chunk.text,
                positive=impl.text,
                task="code2code",
                source="tests",
                language=chunk.language,
                repo=repo,
                id=f"{repo}:{chunk.chunk_id}",
            )
        )

    calls: list[TrainingPair] = []
    call_re = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]{3,})\s*\(")
    for source, chunk in impl_chunks:
        if len(calls) >= max_call_pairs:
            break
        code_lines = chunk.code.splitlines()
        for i, line in enumerate(code_lines[1:], 1):  # skip the definition line itself
            match = next((m for m in call_re.finditer(line) if m.group(1) in unique_defs), None)
            if match is None:
                continue
            target = unique_defs[match.group(1)]
            if target.chunk_id == chunk.chunk_id:
                continue
            context = "\n".join(code_lines[max(0, i - 3) : i + 4])
            calls.append(
                TrainingPair(
                    query=chunker.make_chunk(
                        source, chunk.start_line + max(0, i - 3), chunk.start_line + i + 3, "context", chunk.symbol, context
                    ).text,
                    positive=target.text,
                    task="navigation",
                    source="calls",
                    language=source.language,
                    repo=repo,
                    id=f"{repo}:{chunk.chunk_id}:L{i}",
                )
            )
            break
    return SnapshotPairs(docstring=docstring, tests=tests, calls=calls, undocumented=undocumented)


# --------------------------------------------------------------------------------- commit mining


@dataclass(frozen=True)
class CommitInfo:
    sha: str
    parent: str
    date: str
    author: str
    subject: str
    body: str
    modified: tuple[str, ...] = ()  # paths with status M (renames disabled)

    @property
    def message(self) -> str:
        body_lines = [line for line in self.body.splitlines() if not TRAILER.match(line.strip())]
        body = "\n".join(body_lines).strip()
        subject = PR_REF.sub("", self.subject).strip()
        return f"{subject}\n\n{body}".strip() if body else subject


def _git(repo: str | Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, check=True
    ).stdout.decode("utf-8", errors="replace")


def iter_commits(repo: str | Path, since: str | None = None, until: str | None = None, rev: str = "HEAD") -> Iterator[CommitInfo]:
    """Non-merge commits with exactly one (available) parent, with their modified files, in one `git log`."""
    args = [
        "-c", "core.quotePath=false", "log", "--no-merges", "--no-renames", "--name-status",
        "--format=%x1e%H%x00%P%x00%aI%x00%ae%x00%s%x00%b%x1f", rev,
    ]
    if since:
        args.append(f"--since={since}")
    if until:
        args.append(f"--until={until}")
    for record in _git(repo, *args).split("\x1e"):
        if not record.strip():
            continue
        meta, _, names = record.partition("\x1f")
        sha, parents, date, author, subject, body = meta.split("\x00", 5)
        parents = parents.split()
        if len(parents) != 1:  # merges, roots, and shallow-clone boundaries
            continue
        modified = tuple(
            line.split("\t", 1)[1] for line in names.splitlines() if line.startswith("M\t")
        )
        yield CommitInfo(sha, parents[0], date, author, subject, body.strip(), modified)


_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,(\d+))? @@")


@dataclass
class OldSide:
    """Pre-change view of one file's diff: replaced/deleted old lines, and old lines after which text was inserted."""

    modified: set[int] = field(default_factory=set)
    inserted_after: set[int] = field(default_factory=set)


def diff_old_side(repo: str | Path, parent: str, sha: str, paths: Iterable[str]) -> dict[str, OldSide]:
    """Per path, the pre-change lines a diff touches.

    File headers are only recognised outside hunks (hunk bodies are consumed by their line counts), so a
    deleted line that itself starts with "-- " cannot be mistaken for a header.
    """
    paths = list(paths)
    if not paths:
        return {}
    diff = _git(
        repo, "-c", "core.quotePath=false", "diff", "-U0", "--no-color", "--no-ext-diff", "--no-renames",
        "--src-prefix=a/", "--dst-prefix=b/", parent, sha, "--", *paths,
    )
    out: dict[str, OldSide] = defaultdict(OldSide)
    current: str | None = None
    old_left = new_left = 0
    for line in diff.split("\n"):
        if old_left > 0 or new_left > 0:
            if line.startswith("-"):
                old_left -= 1
            elif line.startswith("+"):
                new_left -= 1
            elif line.startswith(" "):
                old_left -= 1
                new_left -= 1
            continue  # "\\ No newline at end of file" and hunk content
        if line.startswith("--- "):
            name = line[4:].rstrip("\t")  # git appends a tab to paths containing spaces
            current = name[2:] if name.startswith("a/") else None  # /dev/null -> added file
        elif current is not None and (m := _HUNK.match(line)):
            start, count = int(m.group(1)), int(m.group(2) if m.group(2) is not None else 1)
            new_count = int(m.group(3) if m.group(3) is not None else 1)
            old_left, new_left = count, new_count
            if count == 0:
                out[current].inserted_after.add(start)  # "-a,0": insertion after old line a
            else:
                out[current].modified.update(range(start, start + count))
    return dict(out)


def changed_old_lines(repo: str | Path, parent: str, sha: str, paths: Iterable[str]) -> dict[str, set[int]]:
    """Per path: old lines modified or deleted, plus the old lines around each pure insertion."""
    return {
        p: side.modified | {x for a in side.inserted_after for x in (max(1, a), a + 1)}
        for p, side in diff_old_side(repo, parent, sha, paths).items()
    }


def is_useful_commit(c: CommitInfo, min_chars: int = 20) -> bool:
    if "[bot]" in c.author or c.author.startswith(("bot@", "noreply@")) or "dependabot" in c.author:
        return False
    if TRIVIAL_SUBJECT.match(c.subject.strip()):
        return False
    return len(c.message) >= min_chars


def _smallest_chunk(chunks: list[Chunk], line: int) -> Chunk | None:
    """Smallest chunk containing `line`; class skeletons only when nothing finer covers it (e.g. attributes)."""
    containing = [c for c in chunks if c.start_line <= line <= c.end_line]
    return min(containing, key=lambda c: (c.kind == "skeleton", c.end_line - c.start_line), default=None)


def reachable_chunks(chunks: list[Chunk], side: OldSide) -> dict[str, Chunk]:
    """Pre-change chunks the diff edits. An insertion only counts when it lands *inside* a definition body;
    an insertion between definitions (or between class members) creates new code, which no retriever can
    find in the pre-change snapshot, so it is not attributed to a neighbour."""
    touched = {c.chunk_id: c for line in sorted(side.modified) if (c := _smallest_chunk(chunks, line)) is not None}
    for after in sorted(side.inserted_after):
        inside = [c for c in chunks if c.start_line <= after < c.end_line and c.kind not in ("skeleton", "module")]
        if inside:
            c = min(inside, key=lambda c: c.end_line - c.start_line)
            touched[c.chunk_id] = c
    return touched


DOCS_PATH = re.compile(
    r"(^|/)(docs?|documentation|examples?|benchmarks?|bench|website|site|changelog\.d|changes|news|\.github)/", re.I
)


def change_category(path: str, languages: frozenset[str]) -> str:
    """source | test | docs | non_source -- only `source` files can be gold."""
    if detect_language(path) not in languages or _skipped_dir(path):
        # changelogs, lockfiles, configs, declaration files, other languages, and anything the indexer skips
        # (build/, vendored code, ...), which no retriever could return.
        return "non_source"
    if is_test_path(path):
        return "test"
    if DOCS_PATH.search(path):
        return "docs"
    return "source"


def _symbol_names(source: SourceFile) -> set[str]:
    return {s.qualified_name for s in parse_symbols(source)}


@dataclass
class GoldLabels:
    gold_files: list[str]
    chunks: list[Chunk]  # reachable gold chunks in the base snapshot
    created_symbols: list[str]  # "path::Name" defined by the change (absent from base)
    added_source_files: list[str]
    excluded: dict[str, int]  # changed files left out of the gold set, by reason
    file_chunks: list[Chunk] = field(default_factory=list)  # every chunk of the gold files (base snapshot)

    @property
    def gold_symbols(self) -> list[str]:
        return sorted({f"{c.path}::{c.symbol}" for c in self.chunks if c.symbol})

    def meta(self) -> dict:
        n_sym, n_created = len(self.gold_symbols), len(self.created_symbols)
        n_files, n_added = len(self.gold_files), len(self.added_source_files)
        return {
            "excluded_changed_files": self.excluded,
            "added_source_files": self.added_source_files,
            "created_symbols": self.created_symbols[:50],
            "num_created_symbols": n_created,
            # Best achievable recall given that new code cannot be retrieved from the base snapshot.
            "file_recall_ceiling": n_files / (n_files + n_added) if n_files + n_added else 0.0,
            "symbol_recall_ceiling": n_sym / (n_sym + n_created) if n_sym + n_created else 0.0,
        }


def gold_from_diff(
    repo_dir: str | Path,
    base: str,
    head: str,
    languages: frozenset[str],
    chunker: Chunker,
    max_source_files: int,
) -> GoldLabels | str:
    """Gold labels for the change base..head, or the reason the change is unusable."""
    out = _git(repo_dir, "-c", "core.quotePath=false", "diff", "--name-status", "--no-renames", base, head)
    excluded: Counter = Counter()
    modified, added = [], []
    for line in out.splitlines():
        status, _, path = line.partition("\t")
        category = change_category(path, languages)
        if category != "source":
            excluded[category] += 1
        elif status.startswith("M"):
            modified.append(path)
        elif status.startswith("A"):
            added.append(path)
        else:
            excluded["deleted_source"] += 1
    if len(modified) + len(added) + excluded["deleted_source"] > max_source_files:
        return "too_many_files"
    sources = {s.path: s for s in iter_source_files_at_commit_paths(repo_dir, base, modified, languages)}
    for path in modified:
        if path in sources and is_probably_generated(sources[path].text):
            del sources[path]
            excluded["generated"] += 1
    sides = diff_old_side(repo_dir, base, head, list(sources))
    gold, chunks, file_chunks = [], [], []
    for path in modified:
        side = sides.get(path)
        if path not in sources or side is None:
            continue
        gold.append(path)
        all_chunks = chunker.chunk(sources[path])
        file_chunks.extend(all_chunks)
        chunks.extend(reachable_chunks(all_chunks, side).values())
    if not gold:
        return "no_source_gold"
    after = {s.path: s for s in iter_source_files_at_commit_paths(repo_dir, head, gold + added, languages)}
    created = []
    for path in gold:
        if path in after:
            created += [f"{path}::{n}" for n in sorted(_symbol_names(after[path]) - _symbol_names(sources[path]))]
    for path in added:
        if path in after:
            created += [f"{path}::{n}" for n in sorted(_symbol_names(after[path]))]
    excluded["added_source"] = len(added)
    return GoldLabels(gold, chunks, created, sorted(added), {k: v for k, v in sorted(excluded.items()) if v}, file_chunks)


@dataclass
class CommitExample:
    commit: CommitInfo
    gold_files: list[str]
    changed_chunks: list[Chunk]
    labels: GoldLabels | None = None

    def to_case(self, repo: str) -> LocalizationCase:
        symbols = sorted({f"{c.path}::{c.symbol}" for c in self.changed_chunks if c.symbol})
        return LocalizationCase(
            id=f"{repo}@{self.commit.sha[:12]}",
            repo=repo,
            commit=self.commit.parent,
            query=self.commit.message,
            gold_files=self.gold_files,
            gold_symbols=symbols,
            created_at=self.commit.date,
            source="commits",
            meta=self.labels.meta() if self.labels else {},
        )

    def to_pairs(self, repo: str, max_positives: int = 3) -> list[TrainingPair]:
        seen, pairs = set(), []
        for chunk in self.changed_chunks:
            if chunk.chunk_id in seen or len(pairs) >= max_positives:
                continue
            seen.add(chunk.chunk_id)
            pairs.append(
                TrainingPair(
                    query=self.commit.message,
                    positive=chunk.text,
                    task="issue2code",
                    source="commits",
                    language=chunk.language,
                    repo=repo,
                    id=f"{repo}@{self.commit.sha[:12]}:{chunk.chunk_id}",
                    meta={"date": self.commit.date},
                )
            )
        return pairs


def candidate_commits(
    repo_dir: str | Path,
    since: str | None = None,
    until: str | None = None,
    max_gold_files: int = 5,
    exclude_tests: bool = True,
    languages: Iterable[str] = DEFAULT_LANGUAGES,
) -> list[tuple[CommitInfo, list[str]]]:
    """Cheap filter (message + file list only): commits modifying 1..max_gold_files source files."""
    languages = frozenset(languages)
    out = []
    for commit in iter_commits(repo_dir, since, until):
        # `git log --since` filters on committer date; rebased/cherry-picked commits keep an older author date,
        # and the message may have been public since then. Require both dates to be in the window.
        if (since and commit.date[:10] < since) or (until and commit.date[:10] >= until):
            continue
        if not is_useful_commit(commit):
            continue
        files = [f for f in commit.modified if detect_language(f) in languages and not (exclude_tests and is_test_path(f))]
        if files and len(files) <= max_gold_files:
            out.append((commit, files))
    return out


def mine_commits(
    repo_dir: str | Path,
    since: str | None = None,
    until: str | None = None,
    max_gold_files: int = 5,
    exclude_tests: bool = True,
    languages: Iterable[str] = DEFAULT_LANGUAGES,
    chunker: Chunker | None = None,
    sample: int | None = None,
    seed: int = 13,
) -> Iterator[CommitExample]:
    """Commits that modify 1..max_gold_files existing source files, with the pre-change chunks they touch.

    With `sample`, candidates are visited in a seeded random order until `sample` examples are produced,
    so the expensive diff/chunk step only runs for what is kept.
    """
    chunker = chunker or Chunker()
    languages = frozenset(languages)
    candidates = candidate_commits(repo_dir, since, until, max_gold_files, exclude_tests, languages)
    if sample is not None:
        random.Random(seed).shuffle(candidates)
    produced = 0
    for commit, _files in candidates:
        if sample is not None and produced >= sample:
            return
        labels = gold_from_diff(repo_dir, commit.parent, commit.sha, languages, chunker, max_gold_files)
        if isinstance(labels, str):
            continue
        yield CommitExample(commit=commit, gold_files=labels.gold_files, changed_chunks=labels.chunks, labels=labels)
        produced += 1


def iter_source_files_at_commit_paths(
    repo_dir: str | Path, commit: str, paths: Iterable[str], languages: Iterable[str]
) -> Iterator[SourceFile]:
    """Like iter_source_files_at_commit, restricted to `paths` (avoids reading the whole tree)."""
    languages = frozenset(languages)
    for path in sorted(set(paths)):
        try:
            text = _git(repo_dir, "show", f"{commit}:{path}")
        except subprocess.CalledProcessError:
            continue
        lang = detect_language(path)
        if lang in languages:
            yield SourceFile(path=path, language=lang, text=text)


def dedupe_pairs(pairs: Iterable[TrainingPair]) -> list[TrainingPair]:
    from codeintel.evaluation.contamination import exact_key

    seen, out = set(), []
    for p in pairs:
        key = (exact_key(p.query), exact_key(p.positive))
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def snapshot_files(repo_dir: str | Path, rev: str = "HEAD", languages: Iterable[str] = DEFAULT_LANGUAGES) -> list[SourceFile]:
    return list(iter_source_files_at_commit(repo_dir, rev, languages))



# ------------------------------------------------------------------------------- issue -> PR mining

_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)|<img\b[^>]*>", re.I)
_CHECKBOX = re.compile(r"(?m)^\s*[-*]\s*\[[ xX]\].*$\n?")
MAX_ISSUE_CHARS = 6000


def clean_issue_text(title: str, body: str) -> str:
    """Issue title + body without template comments, images, and checkbox boilerplate."""
    body = _IMAGE.sub("", _HTML_COMMENT.sub("", body or ""))
    body = _CHECKBOX.sub("", body)
    body = re.sub(r"\n{3,}", "\n\n", body.replace("\r\n", "\n")).strip()
    text = f"{title.strip()}\n\n{body}".strip()
    return text[:MAX_ISSUE_CHARS]


_IDENT = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
_DOTTED = re.compile(r"`([^`\n]{1,200})`|([A-Za-z_$][A-Za-z0-9_$]*(?:\.[A-Za-z_$][A-Za-z0-9_$]*)*)(\s*\()?")
_CAMEL = re.compile(r"[a-z0-9][A-Z]")
_NOT_IDENTIFIERS = frozenset(
    "self cls this None True False null undefined __init__ __name__ __main__ constructor prototype".split()
)
_GENERIC_BASENAMES = frozenset({"__init__.py", "index.ts", "index.js", "index.tsx", "mod.ts"})
_FILE_EXTENSIONS = frozenset({"py", "pyi", "ts", "tsx", "js", "jsx", "mjs", "cjs", "mts", "cts", "json", "md", "toml", "yaml", "yml"})


def _code_like(token: str) -> bool:
    core = token.strip("_$")
    return "_" in core or bool(_CAMEL.search(token)) or token.startswith("__")


def identifier_doc_freq(chunks: Iterable[Chunk]) -> tuple[Counter, int]:
    """Number of chunks each identifier occurs in, plus the chunk count (for distinctiveness checks)."""
    df: Counter = Counter()
    n = 0
    for c in chunks:
        df.update(set(_IDENT.findall(c.code)))
        n += 1
    return df, n


def query_mentions(
    query: str,
    gold_files: Iterable[str],
    gold_chunks: Iterable[Chunk],
    doc_freq: Mapping[str, int] | None = None,
    num_chunks: int = 0,
    max_df_ratio: float = 0.02,
) -> list[str]:
    """Distinctive identifiers from `gold_chunks`, and gold file paths, that the query names literally.

    Identifiers: a token from the gold code, at least 3 characters, that the query writes as code
    (backticked, followed by "(", snake_case, camelCase, dotted) or that is a PascalCase gold definition
    name. It must also be distinctive in the repository: present in at most `max_df_ratio` of all chunks
    (minimum 3). Measured ubiquitous tokens (`app`, `json`, `config`, `request`) sit at 5-20% of chunks,
    distinctive API names (`parseBody`, `upgradeWebSocket`) below 1.5%. Plain lowercase words are treated
    as English.
    Paths: a gold file's basename (except __init__.py/index.*), a 2+ segment path suffix, or its dotted
    Python module path.
    """
    chunks = list(gold_chunks)
    vocab = {t for c in chunks for t in _IDENT.findall(c.code)} - _NOT_IDENTIFIERS
    symbol_names = {part for c in chunks if c.symbol for part in c.symbol.split(".")}
    limit = max(3, int(max_df_ratio * num_chunks))

    def distinctive(t: str) -> bool:
        return len(t) >= 3 and (doc_freq is None or doc_freq.get(t, 0) <= limit)

    text = re.sub(r"```[\w+-]*", "\n", query)  # code fences, including their language tag
    found: set[str] = set()
    for m in _DOTTED.finditer(text):
        backticked, token, paren = m.group(1), m.group(2), m.group(3)
        if backticked is not None:
            # `cache` names an identifier; `pytest --version` or `--override-ini` only contain English words.
            whole = backticked.strip().rstrip("()")
            for t in _IDENT.findall(backticked):
                if t in vocab and distinctive(t) and (_code_like(t) or t == whole or (t[:1].isupper() and t in symbol_names)):
                    found.add(t)
            continue
        parts = token.split(".")
        if len(parts) > 1 and parts[-1].lower() in _FILE_EXTENSIONS:
            continue  # a filename such as `compress.ts`: handled by the path rules below
        dotted = len(parts) > 1 and all(len(p) >= 2 for p in parts)
        for p in parts:
            if p not in vocab or not distinctive(p):
                continue
            if _code_like(p) or dotted or (paren and p == parts[-1]) or (p in symbol_names and p[:1].isupper()):
                found.add(p)
    lowered = query.lower()
    for path in gold_files:
        segments = path.split("/")
        base = segments[-1]
        stem_path = path.rsplit(".", 1)[0]
        candidates = [] if base in _GENERIC_BASENAMES else [base]
        if len(segments) >= 2:
            candidates.append("/".join(segments[-2:]))
            candidates.append("/".join(stem_path.split("/")[-2:]))
        if path.endswith((".py", ".pyi")) and len(segments) >= 2:
            candidates.append(".".join(stem_path.split("/")[-2:]))
        for cand in candidates:
            if re.search(rf"(?<![\w.-]){re.escape(cand.lower())}(?![\w-])", lowered):
                found.add(path)
                break
    return sorted(found)


def _diff_file_count(repo_dir: str | Path, base: str, head: str) -> int:
    out = _git(repo_dir, "-c", "core.quotePath=false", "diff", "--name-only", "-M", base, head)
    return len([line for line in out.splitlines() if line])


def _has_commit(repo_dir: str | Path, rev: str) -> bool:
    return subprocess.run(["git", "-C", str(repo_dir), "cat-file", "-e", f"{rev}^{{commit}}"], capture_output=True).returncode == 0


def resolve_pr_base(repo_dir: str | Path, merge_commit: str, merge_parents: int, pr_commits: int, changed_files: int) -> str | None:
    """The commit the PR's changes apply to, for merge, squash, and rebase merges.

    Merge commit: first parent. Squash: the single parent. Rebase of k commits: the k-th ancestor. The
    choice is accepted only if the diff touches as many files as GitHub reports for the PR.
    """
    if not _has_commit(repo_dir, merge_commit):
        return None
    if merge_parents == 2:
        return _git(repo_dir, "rev-parse", f"{merge_commit}^1").strip()
    candidates = [f"{merge_commit}^"]
    if pr_commits > 1:
        candidates.append(f"{merge_commit}~{pr_commits}")
    for cand in candidates:
        if _has_commit(repo_dir, cand) and _diff_file_count(repo_dir, cand, merge_commit) == changed_files:
            return _git(repo_dir, "rev-parse", cand).strip()
    return None


@dataclass
class IssueCaseStats:
    links: int = 0
    multi_pr_issues: int = 0
    too_early: int = 0
    after_merge: int = 0
    edited_unrecoverable: int = 0
    bot_or_short: int = 0
    unresolved_range: int = 0
    too_many_files: int = 0
    no_source_gold: int = 0
    kept: int = 0
    kept_symbol_unreachable: int = 0  # kept for file level, but every changed symbol is new code


def mine_issue_cases(
    repo_dir: str | Path,
    repo: str,
    links: Iterable,
    since: str,
    max_source_files: int = 10,
    languages: Iterable[str] = DEFAULT_LANGUAGES,
    chunker: Chunker | None = None,
    min_query_chars: int = 40,
    labels_out: dict[str, GoldLabels] | None = None,
) -> tuple[list[LocalizationCase], IssueCaseStats]:
    """LocalizationCases from issues closed by merged PRs.

    Query: the issue title and body *as opened* (no comments, no later edits). Gold: non-test, non-doc,
    non-generated source files the PR modified, and the pre-existing definitions it edited, all in the PR's
    base snapshot. PRs changing more than `max_source_files` source files are skipped. Definitions and
    files the PR created are unreachable from the base snapshot; they are excluded from the labels and
    recorded in `meta` (with the resulting recall ceilings). Pass `labels_out` to also receive each case's
    GoldLabels (chunk texts), e.g. to turn cases into training pairs with `issue_training_pairs`.
    """
    chunker = chunker or Chunker()
    languages = frozenset(languages)
    links = list(links)
    stats = IssueCaseStats(links=len(links))
    per_issue: dict[int, list] = defaultdict(list)
    for link in links:
        per_issue[link.issue_number].append(link)
    cases: list[LocalizationCase] = []
    doc_freq: Counter | None = None
    n_chunks = 0
    for number, group in sorted(per_issue.items()):
        if len({link.pr_number for link in group}) > 1:
            stats.multi_pr_issues += 1  # ambiguous gold
            continue
        link = group[0]
        if link.issue_created_at[:10] < since:
            stats.too_early += 1
            continue
        if link.issue_created_at > link.pr_merged_at:
            stats.after_merge += 1
            continue
        if link.original_body is None:
            stats.edited_unrecoverable += 1
            continue
        query = clean_issue_text(link.original_title, link.original_body)
        if "[bot]" in link.issue_author or link.issue_author.endswith("bot") or len(query) < min_query_chars:
            stats.bot_or_short += 1
            continue
        base = resolve_pr_base(repo_dir, link.merge_commit, link.merge_parents, link.pr_commits, link.pr_changed_files)
        if base is None:
            stats.unresolved_range += 1
            continue
        labels = gold_from_diff(repo_dir, base, link.merge_commit, languages, chunker, max_source_files)
        if isinstance(labels, str):
            setattr(stats, labels, getattr(stats, labels) + 1)
            continue
        if doc_freq is None:
            # Distinctiveness is judged against one snapshot per repository (the first case's base).
            doc_freq, n_chunks = identifier_doc_freq(
                c for f in iter_source_files_at_commit(repo_dir, base, languages) for c in chunker.chunk(f)
            )
        # lexical: names the edited definitions or a gold path; same-file: names only other code in a gold file
        # (a lexical retriever then finds the file, not the symbol); descriptive: names none of it.
        mentions = query_mentions(query, labels.gold_files, labels.chunks, doc_freq, n_chunks)
        nearby = [] if mentions else query_mentions(query, labels.gold_files, labels.file_chunks, doc_freq, n_chunks)
        tag = "lexical" if mentions else ("same-file" if nearby else "descriptive")
        stats.kept += 1
        stats.kept_symbol_unreachable += not labels.gold_symbols
        if labels_out is not None:
            labels_out[f"{repo}#{number}"] = labels
        cases.append(
            LocalizationCase(
                id=f"{repo}#{number}",
                repo=repo,
                commit=base,
                query=query,
                gold_files=labels.gold_files,
                gold_symbols=labels.gold_symbols,
                created_at=link.issue_created_at,
                source="issues",
                tags={"mentions": tag},
                meta={
                    "pr": link.pr_number,
                    "merge_commit": link.merge_commit,
                    "issue_text": link.body_source,
                    "mentioned": mentions or nearby,
                    **labels.meta(),
                },
            )
        )
    return cases, stats


def issue_training_pairs(case: LocalizationCase, labels: GoldLabels, max_positives: int = 3) -> list[TrainingPair]:
    """issue2code pairs: the issue as opened -> each pre-existing definition its fix edited (base snapshot)."""
    seen, pairs = set(), []
    for chunk in sorted(labels.chunks, key=lambda c: (c.kind == "skeleton", c.chunk_id)):
        if chunk.chunk_id in seen or chunk.symbol is None or len(pairs) >= max_positives:
            continue
        seen.add(chunk.chunk_id)
        pairs.append(
            TrainingPair(
                query=case.query,
                positive=chunk.text,
                task="issue2code",
                source="issues",
                language=chunk.language,
                repo=case.repo,
                id=f"{case.id}:{chunk.chunk_id}",
                meta={"date": case.created_at, "pr": case.meta.get("pr"), "base": case.commit, "mentions": case.tags.get("mentions")},
            )
        )
    return pairs
