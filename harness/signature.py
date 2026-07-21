from __future__ import annotations

import ast
import collections

# Các hàm "bao ngoài" hay gặp trong assert của MBPP — KHÔNG phải hàm cần kiểm thử.
_WRAPPERS = frozenset(
    {
        "abs", "all", "any", "len", "set", "sorted", "list", "tuple", "dict",
        "round", "min", "max", "sum", "str", "int", "float", "bool", "range",
        "frozenset", "isclose", "math", "map", "filter", "zip", "type",
        "repr", "format", "print",
    }
)


def entry_function_name(test_list: list[str]) -> str | None:
    """Suy ra tên hàm cần kiểm thử từ danh sách assert của MBPP.

    Heuristic: đếm mọi lời gọi hàm dạng `name(...)` trong các assert (bỏ qua các
    hàm bao ngoài như set/sorted/round...), trả về tên xuất hiện ở NHIỀU assert
    nhất — đó gần như chắc chắn là hàm entry-point (vì mọi test đều gọi nó).
    """
    counts: collections.Counter[str] = collections.Counter()
    for test in test_list:
        try:
            tree = ast.parse(test.strip())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                name = node.func.id
                if name not in _WRAPPERS:
                    counts[name] += 1
    if not counts:
        return None
    # tie-break ổn định: ưu tiên tần suất cao, rồi tên ngắn hơn, rồi alphabet
    return min(counts, key=lambda n: (-counts[n], len(n), n))
