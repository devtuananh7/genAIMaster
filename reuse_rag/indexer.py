"""
reuse_rag/indexer.py
====================
AST indexer cho một package Python — phần "self-written indexer" của RAG.

Quét toàn bộ file .py của package (mặc định: boltons), dùng module `ast` bóc từng
hàm/lớp ở cấp module thành một CHUNK có cấu trúc:

    {
      "fqn":        "boltons.iterutils.chunked",   # định danh đầy đủ (khoá tra cứu)
      "module":     "boltons.iterutils",
      "name":       "chunked",
      "kind":       "function" | "class",
      "signature":  "def chunked(src, size, count=None, **kw):",
      "docstring":  "<phần prose, đã bỏ doctest>",
      "example":    ">>> chunked(range(10), 3)",   # dòng ví dụ đầu tiên (nếu có)
      "body":       "<toàn bộ source của hàm/lớp>",
      "file":       "boltons/iterutils.py",
      "lineno":     123
    }

Kết quả ghi ra data/boltons_index.json — đây là "vector-store-free" index: numpy +
cosine là bước tuỳ chọn cho thí nghiệm RETRIEVAL; thí nghiệm REPRESENTATION dùng
oracle nên chỉ cần tra theo fqn.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> str:
    """Dựng lại dòng chữ ký từ AST (không kèm thân hàm)."""
    if isinstance(node, ast.ClassDef):
        bases = ", ".join(ast.unparse(base) for base in node.bases)
        return f"class {node.name}({bases}):" if bases else f"class {node.name}:"
    args = ast.unparse(node.args)
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    return f"{prefix} {node.name}({args}):"


def _split_docstring(raw: str | None) -> tuple[str, str]:
    """
    Tách docstring thành (prose, example).
    prose   = các dòng trước doctest đầu tiên (mô tả thuần).
    example = dòng doctest '>>>' đầu tiên KÈM dòng kết quả ngay sau (nếu có),
              tạo thành cặp input→output cụ thể cho mức L3.
    """
    if not raw:
        return "", ""
    lines = raw.splitlines()
    prose_lines: list[str] = []
    example_lines: list[str] = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(">>>"):
            example_lines.append(stripped)
            # gộp các dòng nối tiếp '...' và MỘT dòng kết quả (không phải '>>>').
            for nxt in lines[idx + 1:]:
                nstr = nxt.strip()
                if nstr.startswith("..."):
                    example_lines.append(nstr)
                    continue
                if nstr and not nstr.startswith(">>>"):
                    example_lines.append(nstr)  # dòng kết quả mong đợi
                break
            break
        prose_lines.append(line.rstrip())
    prose = "\n".join(prose_lines).strip()
    example = "\n".join(example_lines).strip()
    return prose, example


def index_source(source: str, module: str, rel_file: str) -> dict[str, dict[str, Any]]:
    """Bóc các hàm/lớp cấp module trong một file nguồn thành chunk."""
    chunks: dict[str, dict[str, Any]] = {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return chunks

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            kind = "function"
        elif isinstance(node, ast.ClassDef):
            kind = "class"
        else:
            continue
        if node.name.startswith("_"):
            continue  # bỏ qua private/dunder

        prose, example = _split_docstring(ast.get_docstring(node, clean=True))
        fqn = f"{module}.{node.name}"
        chunks[fqn] = {
            "fqn": fqn,
            "module": module,
            "name": node.name,
            "kind": kind,
            "signature": _signature(node),
            "docstring": prose,
            "example": example,
            "body": ast.get_source_segment(source, node) or "",
            "file": rel_file,
            "lineno": node.lineno,
        }
    return chunks


def _module_name(package: str, package_dir: Path, py_file: Path) -> str:
    rel = py_file.relative_to(package_dir).with_suffix("")
    parts = [package, *[p for p in rel.parts if p != "__init__"]]
    return ".".join(parts)


def build_index(package: str = "boltons") -> dict[str, dict[str, Any]]:
    """Import package để tìm thư mục nguồn, rồi index mọi file .py."""
    import importlib

    mod = importlib.import_module(package)
    package_dir = Path(mod.__file__).resolve().parent

    index: dict[str, dict[str, Any]] = {}
    for py_file in sorted(package_dir.rglob("*.py")):
        module = _module_name(package, package_dir, py_file)
        source = py_file.read_text(encoding="utf-8", errors="replace")
        rel_file = str(py_file.relative_to(package_dir.parent))
        index.update(index_source(source, module, rel_file))
    return index


def load_index(path: str | Path) -> dict[str, dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_index(path: str | Path, *, package: str = "boltons") -> dict[str, dict[str, Any]]:
    index = build_index(package)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(index, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AST chunk index for a Python package.")
    parser.add_argument("--package", default="boltons")
    parser.add_argument("--output", default="data/boltons_index.json")
    args = parser.parse_args()

    index = write_index(args.output, package=args.package)
    print(f"indexed {len(index)} symbols from {args.package} -> {args.output}")


if __name__ == "__main__":
    main()
