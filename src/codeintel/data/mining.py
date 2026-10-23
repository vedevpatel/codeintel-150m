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
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from codeintel.data.schemas import LocalizationCase, TrainingPair
from codeintel.indexing.chunker import Chunker
from codeintel.indexing.parser import parse_symbols
from codeintel.indexing.repository import DEFAULT_LANGUAGES, detect_language, iter_source_files_at_commit
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


def changed_old_lines(repo: str | Path, parent: str, sha: str, paths: Iterable[str]) -> dict[str, set[int]]:
    """Per path: lines at `parent` that the commit modified or deleted (pure insertions map to adjacent lines).

    File headers are only recognised outside hunks (hunk bodies are consumed by their line counts), so a
    deleted line that itself starts with "-- " cannot be mistaken for a header.
    """
    diff = _git(
        repo, "-c", "core.quotePath=false", "diff", "-U0", "--no-color", "--no-ext-diff", "--no-renames",
        "--src-prefix=a/", "--dst-prefix=b/", parent, sha, "--", *paths,
    )
    out: dict[str, set[int]] = defaultdict(set)
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
            continue  # "\ No newline at end of file" and hunk content
        if line.startswith("--- "):
            name = line[4:].rstrip("\t")  # git appends a tab to paths containing spaces
            current = name[2:] if name.startswith("a/") else None  # /dev/null -> added file
        elif current is not None and (m := _HUNK.match(line)):
            start, count = int(m.group(1)), int(m.group(2) if m.group(2) is not None else 1)
            new_count = int(m.group(3) if m.group(3) is not None else 1)
            old_left, new_left = count, new_count
            if count == 0:
                out[current].update({max(1, start), start + 1})
            else:
                out[current].update(range(start, start + count))
    return out


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


@dataclass
class CommitExample:
    commit: CommitInfo
    gold_files: list[str]
    changed_chunks: list[Chunk]

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
    for commit, files in candidates:
        if sample is not None and produced >= sample:
            return
        sources = {s.path: s for s in iter_source_files_at_commit_paths(repo_dir, commit.parent, files, languages)}
        changed_lines = changed_old_lines(repo_dir, commit.parent, commit.sha, files)
        gold, changed = [], []
        for path in files:
            source = sources.get(path)
            if source is None:
                continue
            lines = changed_lines.get(path)
            if not lines:
                continue
            # Every modified file is gold, even if the change falls between chunks (e.g. blank-line edits).
            gold.append(path)
            chunks = chunker.chunk(source)
            touched = {c.chunk_id: c for line in sorted(lines) if (c := _smallest_chunk(chunks, line)) is not None}
            changed.extend(touched.values())
        if not gold:
            continue
        yield CommitExample(commit=commit, gold_files=gold, changed_chunks=changed)
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

