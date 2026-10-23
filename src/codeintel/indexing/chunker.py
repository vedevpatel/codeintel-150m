"""Split source files into retrieval chunks aligned with definitions.

Per file:
  * every leaf definition (function, method, small class, interface, type, ...) becomes one chunk;
  * a class-like definition with members also gets a *skeleton* chunk: its own lines with each member
    body collapsed to its signature (useful for navigation queries about a type as a whole);
  * module-level code between top-level definitions (imports, constants, scripts) becomes module chunks;
  * anything exceeding the size limits is split into overlapping line windows;
  * files without a parser are split into line windows.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass

from codeintel.indexing.parser import parse_symbols
from codeintel.schemas import Chunk, SourceFile, Symbol, split_lines

COMMENT_PREFIX = {"python": "#"}
SKELETON_KINDS = frozenset({"class", "interface", "enum", "namespace"})


@dataclass(frozen=True)
class ChunkerConfig:
    max_lines: int = 120
    max_chars: int = 4000
    window_overlap: int = 16
    min_module_lines: int = 2
    include_path_header: bool = True
    class_skeletons: bool = True


@dataclass
class _Node:
    symbol: Symbol
    children: list[_Node]


def _build_tree(symbols: list[Symbol]) -> list[_Node]:
    """Nest symbols by line containment (robust to languages where scope != lexical nesting, e.g. Rust impls)."""
    roots: list[_Node] = []
    stack: list[_Node] = []
    for sym in sorted(symbols, key=lambda s: (s.start_line, -s.end_line)):
        node = _Node(sym, [])
        while stack and not (stack[-1].symbol.start_line <= sym.start_line and sym.end_line <= stack[-1].symbol.end_line):
            stack.pop()
        (stack[-1].children if stack else roots).append(node)
        stack.append(node)
    return roots


class Chunker:
    def __init__(self, config: ChunkerConfig | None = None):
        self.config = config or ChunkerConfig()

    # -- rendering -------------------------------------------------------------------------------
    def _header(self, source: SourceFile, symbol: str | None, part: str = "") -> str:
        if not self.config.include_path_header:
            return ""
        c = COMMENT_PREFIX.get(source.language, "//")
        label = f"{source.path}::{symbol}" if symbol else source.path
        return f"{c} {label}{part}\n"

    def make_chunk(
        self, source: SourceFile, start: int, end: int, kind: str, symbol: str | None, code: str, part: str = ""
    ) -> Chunk:
        body = textwrap.dedent(code).strip("\n")
        if len(body) > self.config.max_chars:
            body = body[: self.config.max_chars]
        return Chunk(
            chunk_id=f"{source.path}#L{start}-L{end}:{kind}:{symbol or ''}",
            path=source.path,
            language=source.language,
            start_line=start,
            end_line=end,
            kind=kind,
            symbol=symbol,
            text=self._header(source, symbol, part) + body,
            code=code,
        )

    # -- windows ---------------------------------------------------------------------------------
    def _window_ranges(self, lines: list[str], start: int, end: int) -> list[tuple[int, int]]:
        cfg = self.config
        ranges = []
        i = start
        while i <= end:
            j, chars = i, 0
            while j <= end and (j - i) < cfg.max_lines and (j == i or chars + len(lines[j - 1]) + 1 <= cfg.max_chars):
                chars += len(lines[j - 1]) + 1
                j += 1
            ranges.append((i, j - 1))
            if j > end:
                break
            i = max(j - cfg.window_overlap, i + 1)
        return ranges

    def _fits(self, lines: list[str], start: int, end: int) -> bool:
        span = lines[start - 1 : end]
        return len(span) <= self.config.max_lines and sum(len(x) + 1 for x in span) <= self.config.max_chars

    def _emit_span(
        self, source: SourceFile, lines: list[str], start: int, end: int, kind: str, symbol: str | None
    ) -> list[Chunk]:
        if self._fits(lines, start, end):
            return [self.make_chunk(source, start, end, kind, symbol, "\n".join(lines[start - 1 : end]))]
        ranges = self._window_ranges(lines, start, end)
        return [
            self.make_chunk(source, s, e, kind, symbol, "\n".join(lines[s - 1 : e]), part=f" (part {k}/{len(ranges)})")
            for k, (s, e) in enumerate(ranges, 1)
        ]

    # -- structure-aware chunks ------------------------------------------------------------------
    def _skeleton(self, source: SourceFile, lines: list[str], node: _Node) -> Chunk | None:
        sym = node.symbol
        out: list[str] = []
        line = sym.start_line
        own_lines = 0
        for child in node.children:
            c = child.symbol
            own = lines[line - 1 : c.start_line - 1]
            out.extend(own)
            own_lines += sum(1 for x in own if x.strip())
            if c.end_line - c.start_line <= 1:
                out.extend(lines[c.start_line - 1 : c.end_line])
            else:
                first = lines[c.start_line - 1]
                indent = first[: len(first) - len(first.lstrip())]
                stub = " ..." if source.language == "python" else " { ... }"
                out.append(f"{indent}{c.signature}{stub}")
            line = c.end_line + 1
        tail = lines[line - 1 : sym.end_line]
        out.extend(tail)
        own_lines += sum(1 for x in tail if x.strip())
        # A namespace skeleton is only useful if it holds code beyond its members.
        if sym.kind == "namespace" and own_lines <= 2:
            return None
        code = "\n".join(out)
        if len(out) > self.config.max_lines * 2:
            code = "\n".join(out[: self.config.max_lines * 2])
        return self.make_chunk(source, sym.start_line, sym.end_line, "skeleton", sym.qualified_name, code)

    def _chunk_node(self, source: SourceFile, lines: list[str], node: _Node) -> list[Chunk]:
        sym = node.symbol
        if not node.children:
            return self._emit_span(source, lines, sym.start_line, sym.end_line, sym.kind, sym.qualified_name)
        chunks: list[Chunk] = []
        if self.config.class_skeletons and sym.kind in SKELETON_KINDS:
            skeleton = self._skeleton(source, lines, node)
            if skeleton is not None:
                chunks.append(skeleton)
        elif sym.kind not in SKELETON_KINDS:
            # A function with nested definitions (rare outside Python closures): keep it whole.
            return self._emit_span(source, lines, sym.start_line, sym.end_line, sym.kind, sym.qualified_name)
        for child in node.children:
            chunks.extend(self._chunk_node(source, lines, child))
        return chunks

    def _module_chunks(self, source: SourceFile, lines: list[str], roots: list[_Node]) -> list[Chunk]:
        covered = [False] * (len(lines) + 2)
        for node in roots:
            for i in range(node.symbol.start_line, node.symbol.end_line + 1):
                covered[i] = True
        chunks: list[Chunk] = []
        i = 1
        while i <= len(lines):
            if covered[i]:
                i += 1
                continue
            j = i
            while j + 1 <= len(lines) and not covered[j + 1]:
                j += 1
            nonblank = sum(1 for x in lines[i - 1 : j] if x.strip())
            if nonblank >= self.config.min_module_lines or (not roots and nonblank):
                # Trim leading/trailing blank lines so line ranges point at real code.
                s, e = i, j
                while s < e and not lines[s - 1].strip():
                    s += 1
                while e > s and not lines[e - 1].strip():
                    e -= 1
                chunks.extend(self._emit_span(source, lines, s, e, "module", None))
            i = j + 1
        return chunks

    def chunk(self, source: SourceFile, symbols: list[Symbol] | None = None) -> list[Chunk]:
        lines = split_lines(source.text)
        if not lines:
            return []
        if symbols is None:
            symbols = parse_symbols(source)
        roots = _build_tree(symbols)
        chunks = self._module_chunks(source, lines, roots)
        for node in roots:
            chunks.extend(self._chunk_node(source, lines, node))
        chunks.sort(key=lambda c: (c.start_line, c.kind != "skeleton", c.end_line))
        return chunks
