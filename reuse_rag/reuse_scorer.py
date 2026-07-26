"""
reuse_rag/reuse_scorer.py
=========================
Metric HEADLINE: code do model sinh ra có TÁI SỬ DỤNG đúng API đích không?

Chấm bằng AST tĩnh (không cần chạy code). Bắt được 3 dạng tái dùng hợp lệ:

  1. from boltons.iterutils import chunked   →  chunked(...)
  2. from boltons import iterutils           →  iterutils.chunked(...)
  3. import boltons(.iterutils)              →  boltons.iterutils.chunked(...)

CHỐNG DƯƠNG TÍNH GIẢ (điểm quan trọng nhất):
  - Nếu model TỰ ĐỊNH NGHĨA lại `def chunked(...)` rồi gọi `chunked(...)`, đó KHÔNG
    phải tái dùng — chỉ là trùng tên. Bare-name call chỉ tính là reuse khi tên đó
    được IMPORT từ đúng module VÀ không bị định nghĩa lại cục bộ.
  - Gọi qua attribute (iterutils.chunked / boltons.iterutils.chunked) luôn an toàn
    vì đường dẫn phải khớp module đích.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass(frozen=True)
class ReuseResult:
    reused: bool
    imported: bool          # có import binding trỏ tới API đích không
    self_defined: bool      # model có tự định nghĩa lại tên đó không
    detail: str             # mô tả cách phát hiện (debug/log)


def _module_tail(module: str) -> str:
    """boltons.iterutils -> iterutils (đoạn cuối để khớp gọi qua submodule)."""
    return module.rsplit(".", 1)[-1]


def _attribute_path(node: ast.expr) -> list[str]:
    """Trả về chuỗi tên của một attribute/name chain, vd boltons.iterutils.chunked."""
    parts: list[str] = []
    cur: ast.expr | None = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    parts.reverse()
    return parts


def score_reuse(code: str, target_module: str, target_name: str) -> ReuseResult:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ReuseResult(False, False, False, "syntax_error")

    module_tail = _module_tail(target_module)

    # --- Thu thập import binding trỏ tới API đích ---
    direct_name_bindings: set[str] = set()   # tên gọi trực tiếp: chunked
    submodule_bindings: set[str] = set()      # tên submodule: iterutils
    root_module_imported = False              # import boltons / boltons.iterutils

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == target_module:
                for alias in node.names:
                    if alias.name == target_name:
                        direct_name_bindings.add(alias.asname or alias.name)
            # from boltons import iterutils
            if node.module and target_module.startswith(node.module + "."):
                for alias in node.names:
                    if alias.name == module_tail:
                        submodule_bindings.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == target_module or target_module.startswith(alias.name + "."):
                    root_module_imported = True
                # import boltons.iterutils as iu
                if alias.name == target_module and alias.asname:
                    submodule_bindings.add(alias.asname)

    imported = bool(direct_name_bindings or submodule_bindings or root_module_imported)

    # --- Model có tự định nghĩa lại tên đích không? ---
    self_defined = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.name == target_name
        for node in ast.walk(tree)
    )

    # --- Duyệt các lời gọi ---
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func

        # Dạng bare-name: chunked(...)
        if isinstance(func, ast.Name):
            if (
                func.id in direct_name_bindings
                and func.id == target_name
                and not self_defined
            ):
                return ReuseResult(True, imported, self_defined, "bare_name_import")

        # Dạng attribute: iterutils.chunked / boltons.iterutils.chunked / iu.chunked
        elif isinstance(func, ast.Attribute) and func.attr == target_name:
            path = _attribute_path(func)
            # khớp nếu chuỗi kết thúc bằng <module_tail>.<name>
            if len(path) >= 2 and path[-1] == target_name:
                owner = path[-2]
                if owner == module_tail or owner in submodule_bindings:
                    return ReuseResult(True, imported, self_defined, "attribute_call")
                # boltons.iterutils.chunked (full path)
                if ".".join(path[:-1]) == target_module:
                    return ReuseResult(True, imported, self_defined, "full_path_call")

    return ReuseResult(False, imported, self_defined, "no_reuse_call")
