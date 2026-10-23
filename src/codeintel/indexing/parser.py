"""Extract definitions (functions, methods, classes, ...) from source files.

Python uses the standard-library `ast`. Other languages use tree-sitter grammars; a language whose
grammar package is not installed yields no symbols and is chunked by line windows instead.

Symbol spans include decorators and directly-attached leading doc comments.
"""

from __future__ import annotations

import ast
import importlib
import logging
import re
from dataclasses import dataclass, field, replace
from functools import cache

from codeintel.schemas import SourceFile, Symbol, split_lines

logger = logging.getLogger(__name__)

_WS = re.compile(r"\s+")
MAX_SIGNATURE_CHARS = 300


def _clean_signature(text: str) -> str:
    return _WS.sub(" ", text).strip()[:MAX_SIGNATURE_CHARS]


# --------------------------------------------------------------------------------------------- Python


def _python_header(lines: list[str], node: ast.AST) -> str:
    """`def name(args) -> ret:` / `class Name(Base):` text, without decorators or body."""
    first = node.body[0]
    segment = lines[node.lineno - 1 : first.lineno]
    if segment:
        # col_offset counts UTF-8 bytes, not characters.
        segment[-1] = segment[-1].encode("utf-8")[: first.col_offset].decode("utf-8", errors="ignore")
    return _clean_signature(" ".join(segment)).rstrip(":").rstrip() + ":"


def _python_doc_span(node: ast.AST) -> tuple[int, int] | None:
    body = getattr(node, "body", None)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return (body[0].lineno, body[0].end_lineno or body[0].lineno)
    return None


def parse_python(source: SourceFile) -> list[Symbol]:
    try:
        tree = ast.parse(source.text)
    except (SyntaxError, ValueError):
        return []
    lines = split_lines(source.text)
    symbols: list[Symbol] = []

    def visit(body: list[ast.stmt], scope: list[str], in_class: bool) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                is_class = isinstance(node, ast.ClassDef)
                kind = "class" if is_class else ("method" if in_class else "function")
                start = min([node.lineno] + [d.lineno for d in node.decorator_list])
                qualified = ".".join(scope + [node.name])
                symbols.append(
                    Symbol(
                        name=node.name,
                        qualified_name=qualified,
                        kind=kind,
                        start_line=start,
                        end_line=node.end_lineno or node.lineno,
                        signature=_python_header(lines, node),
                        docstring=ast.get_docstring(node) or "",
                        parent=".".join(scope) or None,
                        doc_span=_python_doc_span(node),
                    )
                )
                # Nested functions stay inside their parent's chunk; class bodies are expanded.
                if is_class:
                    visit(node.body, scope + [node.name], in_class=True)
            elif isinstance(node, (ast.If, ast.Try)) and not scope:
                # Definitions guarded by `if TYPE_CHECKING:` / `try: import ... except:` at module level.
                for child_body in (node.body, getattr(node, "orelse", []), getattr(node, "finalbody", [])):
                    visit(child_body, scope, in_class)

    visit(tree.body, [], in_class=False)
    return symbols


# ---------------------------------------------------------------------------------------- tree-sitter


@dataclass(frozen=True)
class TreeSitterSpec:
    module: str
    function: str = "language"
    # node type -> symbol kind
    definitions: dict[str, str] = field(default_factory=dict)
    # node types whose `body` field holds member definitions
    containers: frozenset[str] = frozenset()
    # containers that are not symbols themselves (e.g. Rust `impl`); scope name from the `type` field
    scope_only: frozenset[str] = frozenset()
    # wrappers whose definition child determines the symbol (span starts at the wrapper)
    wrappers: frozenset[str] = frozenset()
    # `const f = () => ...` style declarations
    lexical: frozenset[str] = frozenset()
    declarators: frozenset[str] = frozenset({"variable_declarator"})
    # class fields whose value may be a function (`handler = () => ...`)
    fields: frozenset[str] = frozenset()
    function_values: frozenset[str] = frozenset()
    comments: frozenset[str] = frozenset({"comment"})


_TS_COMMON = dict(
    definitions={
        "function_declaration": "function",
        "generator_function_declaration": "function",
        "function_signature": "function",
        "class_declaration": "class",
        "abstract_class_declaration": "class",
        "class": "class",
        "method_definition": "method",
        "method_signature": "method",
        "abstract_method_signature": "method",
        "interface_declaration": "interface",
        "type_alias_declaration": "type",
        "enum_declaration": "enum",
        "internal_module": "namespace",
        "module": "namespace",
    },
    containers=frozenset(
        {"class_declaration", "abstract_class_declaration", "class", "interface_declaration", "internal_module", "module"}
    ),
    wrappers=frozenset({"export_statement", "expression_statement", "ambient_declaration"}),
    lexical=frozenset({"lexical_declaration", "variable_declaration"}),
    fields=frozenset({"public_field_definition", "field_definition"}),
    function_values=frozenset({"arrow_function", "function_expression", "function", "generator_function"}),
)

TREE_SITTER_SPECS: dict[str, TreeSitterSpec] = {
    "typescript": TreeSitterSpec(module="tree_sitter_typescript", function="language_typescript", **_TS_COMMON),
    "tsx": TreeSitterSpec(module="tree_sitter_typescript", function="language_tsx", **_TS_COMMON),
    # The TSX grammar parses modern JavaScript (including JSX) without a separate grammar package.
    "javascript": TreeSitterSpec(module="tree_sitter_typescript", function="language_tsx", **_TS_COMMON),
    "java": TreeSitterSpec(
        module="tree_sitter_java",
        definitions={
            "class_declaration": "class",
            "interface_declaration": "interface",
            "enum_declaration": "enum",
            "record_declaration": "class",
            "annotation_type_declaration": "interface",
            "method_declaration": "method",
            "constructor_declaration": "method",
        },
        containers=frozenset(
            {"class_declaration", "interface_declaration", "enum_declaration", "record_declaration"}
        ),
        comments=frozenset({"line_comment", "block_comment"}),
    ),
    "go": TreeSitterSpec(
        module="tree_sitter_go",
        definitions={
            "function_declaration": "function",
            "method_declaration": "method",
            "type_spec": "type",
        },
        wrappers=frozenset({"type_declaration"}),
    ),
    "rust": TreeSitterSpec(
        module="tree_sitter_rust",
        definitions={
            "function_item": "function",
            "function_signature_item": "function",
            "struct_item": "class",
            "enum_item": "enum",
            "union_item": "class",
            "trait_item": "interface",
            "type_item": "type",
            "mod_item": "namespace",
            "macro_definition": "function",
        },
        containers=frozenset({"trait_item", "mod_item"}),
        scope_only=frozenset({"impl_item"}),
        comments=frozenset({"line_comment", "block_comment"}),
    ),
}


@cache
def _ts_parser(language: str):
    spec = TREE_SITTER_SPECS.get(language)
    if spec is None:
        return None
    try:
        from tree_sitter import Language, Parser

        module = importlib.import_module(spec.module)
        return Parser(Language(getattr(module, spec.function)()))
    except (ImportError, AttributeError) as exc:
        logger.info("no tree-sitter grammar for %s (%s); falling back to line windows", language, exc)
        return None


def has_parser(language: str) -> bool:
    return language == "python" or _ts_parser(language) is not None


_COMMENT_MARKERS = re.compile(r"^\s*(/\*\*?|\*/|\*|///?|//!|#)\s?")


def _clean_comment(text: str) -> str:
    out = []
    for line in text.splitlines():
        line = _COMMENT_MARKERS.sub("", line).rstrip()
        line = line.removesuffix("*/").rstrip()
        out.append(line)
    return "\n".join(out).strip()


class _TreeSitterExtractor:
    def __init__(self, spec: TreeSitterSpec, src: bytes):
        self.spec = spec
        self.src = src
        self.symbols: list[Symbol] = []

    def text(self, node) -> str:
        return self.src[node.start_byte : node.end_byte].decode("utf-8", errors="replace")

    def name_of(self, node) -> str | None:
        name = node.child_by_field_name("name")
        return self.text(name) if name is not None else None

    def leading_comments(self, node):
        comments = []
        prev = node.prev_named_sibling
        row = node.start_point[0]
        while prev is not None and prev.type in self.spec.comments and prev.end_point[0] >= row - 1:
            comments.append(prev)
            row = prev.start_point[0]
            prev = prev.prev_named_sibling
        return list(reversed(comments))

    def signature(self, node, body_owner=None) -> str:
        owner = body_owner if body_owner is not None else node
        body = owner.child_by_field_name("body")
        end = body.start_byte if body is not None else node.end_byte
        text = self.src[node.start_byte : end].decode("utf-8", errors="replace")
        if body is None:
            text = text.split("\n", 1)[0]
        return _clean_signature(text)

    def emit(self, outer, target, name: str, kind: str, scope: list[str], body_owner=None) -> None:
        comments = self.leading_comments(outer)
        start_node = comments[0] if comments else outer
        doc = "\n".join(_clean_comment(self.text(c)) for c in comments)
        doc_span = (comments[0].start_point[0] + 1, comments[-1].end_point[0] + 1) if comments else None
        self.symbols.append(
            Symbol(
                name=name,
                qualified_name=".".join(scope + [name]),
                kind=kind,
                start_line=start_node.start_point[0] + 1,
                end_line=outer.end_point[0] + 1,
                signature=self.signature(target, body_owner),
                docstring=doc,
                parent=".".join(scope) or None,
                doc_span=doc_span,
            )
        )

    def unwrap(self, node):
        target = node
        while target.type in self.spec.wrappers:
            inner = target.child_by_field_name("declaration")
            if inner is None:
                inner = next(
                    (
                        c
                        for c in target.named_children
                        if c.type in self.spec.definitions
                        or c.type in self.spec.lexical
                        or c.type in self.spec.scope_only
                        or c.type in self.spec.wrappers
                        or c.type in self.spec.function_values
                    ),
                    None,
                )
            if inner is None:
                return target
            target = inner
        return target

    def scope_name(self, node) -> str:
        type_node = node.child_by_field_name("type")
        match = re.match(r"[\w:]+", self.text(type_node)) if type_node is not None else None
        return match.group(0) if match else "impl"

    def visit(self, node, scope: list[str], in_container: bool) -> None:
        for child in node.named_children:
            # Go: `type A struct{}` or a grouped `type ( A ...; B ... )` declaration.
            if child.type == "type_declaration":
                specs = [c for c in child.named_children if c.type in self.spec.definitions]
                for spec_node in specs:
                    outer = child if len(specs) == 1 else spec_node
                    self.handle_definition(outer, spec_node, scope, in_container)
                continue
            target = self.unwrap(child)
            if target.type in self.spec.definitions:
                self.handle_definition(child, target, scope, in_container)
            elif target.type in self.spec.scope_only:
                body = target.child_by_field_name("body")
                if body is not None:
                    self.visit(body, scope + [self.scope_name(target)], in_container=True)
            elif target.type in self.spec.function_values and target is not child:
                # `export default function () {}` / `export default () => ...`
                self.emit(child, target, "default", "method" if in_container else "function", scope)
            elif target.type in self.spec.lexical:
                declarators = [d for d in target.named_children if d.type in self.spec.declarators]
                for decl in declarators:
                    value = decl.child_by_field_name("value")
                    name_node = decl.child_by_field_name("name")
                    if value is None or name_node is None:
                        continue
                    if value.type in self.spec.function_values:
                        kind = "method" if in_container else "function"
                        sig_node = target if len(declarators) == 1 else decl
                        self.emit(child, sig_node, self.text(name_node), kind, scope, body_owner=value)
                    elif value.type in ("class", "class_expression"):
                        self.handle_definition(child, value, scope, in_container, name=self.text(name_node))
            elif target.type in self.spec.fields and in_container:
                value = target.child_by_field_name("value")
                name_node = target.child_by_field_name("name") or target.child_by_field_name("property")
                if value is not None and name_node is not None and value.type in self.spec.function_values:
                    self.emit(child, target, self.text(name_node), "method", scope, body_owner=value)

    def handle_definition(self, outer, target, scope, in_container, name: str | None = None) -> None:
        kind = self.spec.definitions[target.type]
        if kind == "function" and in_container:
            kind = "method"
        name = name or self.name_of(target) or "default"
        if target.type == "method_declaration" and self.spec.module == "tree_sitter_go":
            receiver = target.child_by_field_name("receiver")
            match = re.search(r"\*?\s*([A-Za-z_]\w*)\s*(\[[^\]]*\])?\s*\)\s*$", self.text(receiver)) if receiver else None
            scope = scope + [match.group(1)] if match else scope
        self.emit(outer, target, name, kind, scope)
        if target.type in self.spec.containers:
            body = target.child_by_field_name("body")
            if body is not None:
                # Members of classes/interfaces/traits are methods; members of namespaces/modules are not.
                self.visit(body, scope + [name], in_container=kind != "namespace")


def parse_tree_sitter(source: SourceFile) -> list[Symbol]:
    parser = _ts_parser(source.language)
    if parser is None:
        return []
    src = source.text.encode("utf-8")
    tree = parser.parse(src)
    extractor = _TreeSitterExtractor(TREE_SITTER_SPECS[source.language], src)
    extractor.visit(tree.root_node, [], in_container=False)
    return extractor.symbols


def parse_symbols(source: SourceFile) -> list[Symbol]:
    """Definitions in `source`, ordered by start line. Empty if the language has no parser."""
    symbols = parse_python(source) if source.language == "python" else parse_tree_sitter(source)
    n_lines = max(1, source.text.count("\n") + 1)
    clamped = [
        replace(s, start_line=max(1, s.start_line), end_line=min(n_lines, max(s.start_line, s.end_line)))
        for s in symbols
    ]
    return sorted(clamped, key=lambda s: (s.start_line, -s.end_line))
