import pytest

from codeintel.indexing.parser import has_parser, parse_symbols
from codeintel.schemas import SourceFile

from .conftest import SAMPLE_REPO


def symbols(path: str, language: str, text: str | None = None):
    text = text if text is not None else (SAMPLE_REPO / path).read_text()
    return {s.qualified_name: s for s in parse_symbols(SourceFile(path, language, text))}


def test_python_definitions_and_nesting():
    syms = symbols("inventory/models.py", "python")
    assert {"OutOfStockError", "Item", "Item.is_available", "Inventory", "Inventory.add", "Inventory.low_stock"} <= set(syms)
    assert syms["Inventory"].kind == "class"
    assert syms["Inventory.remove"].kind == "method"
    assert syms["Inventory.remove"].parent == "Inventory"
    assert syms["Inventory.remove"].signature == "def remove(self, sku: str, quantity: int) -> Item:"
    assert syms["Inventory.add"].docstring.startswith("Insert a new item")


def test_python_decorators_docspan_and_one_liners():
    text = (
        "@decorator\n@other(1)\ndef top(a,\n        b=2):\n"
        '    """Doc.\n\n    More."""\n    return a\n\n'
        "class C:\n    def short(self): return 1\n"
    )
    syms = symbols("x.py", "python", text)
    top = syms["top"]
    assert (top.start_line, top.end_line) == (1, 8)
    assert top.doc_span == (5, 7)
    assert top.signature == "def top(a, b=2):"
    assert syms["C.short"].signature == "def short(self):"


def test_python_syntax_error_yields_nothing():
    assert symbols("bad.py", "python", "def broken(:\n  pass\n") == {}


def test_typescript_functions_classes_and_arrows():
    syms = symbols("web/src/api/client.ts", "typescript")
    assert syms["fetchWithRetry"].kind == "function"
    assert syms["fetchWithRetry"].docstring.startswith("Fetch a URL")
    assert syms["fetchWithRetry"].doc_span is not None
    assert syms["ItemDto"].kind == "interface"
    assert syms["ApiClient.getItem"].kind == "method"
    assert syms["ApiClient.getItem"].docstring == "Load a single item by SKU."
    arrows = symbols("web/src/utils/debounce.ts", "typescript")
    assert arrows["throttle"].kind == "function"
    assert arrows["throttle"].signature.startswith("const throttle =")


def test_tsx_default_export_component():
    syms = symbols("web/src/components/Cart.tsx", "tsx")
    assert "formatCents" in syms
    assert "Cart" in syms and syms["Cart"].kind == "function"


def test_namespace_members_are_functions_not_methods():
    syms = symbols("a.ts", "typescript", "namespace NS { export function inner() {} }\n")
    assert syms["NS.inner"].kind == "function"


@pytest.mark.parametrize(
    "language,module,text,expected",
    [
        ("java", "tree_sitter_java", "class Svc {\n  /** ctor */\n  Svc() {}\n  int get() { return 1; }\n}\n", {"Svc", "Svc.Svc", "Svc.get"}),
        ("go", "tree_sitter_go", "package m\ntype S struct{}\nfunc (s *S) Run() {}\nfunc f() {}\n", {"S", "S.Run", "f"}),
        ("rust", "tree_sitter_rust", "struct P;\nimpl P { fn new() -> Self { P } }\nfn free() {}\n", {"P", "P.new", "free"}),
    ],
)
def test_optional_grammars(language, module, text, expected):
    pytest.importorskip(module)
    assert has_parser(language)
    assert expected <= set(symbols(f"x.{language}", language, text))
