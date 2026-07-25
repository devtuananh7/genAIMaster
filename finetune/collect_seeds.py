"""
finetune/collect_seeds.py  — Stage A, bước 1
============================================
Thu 300-500 seed snippet (1-15 dòng) từ các package Python license mở đã cài.
Mỗi seed GHI RÕ NGUỒN (repo/file/dòng) để tuân thủ yêu cầu trích dẫn của spec.

Seed = một hàm ngắn (<=15 dòng) bóc bằng AST. Đây chỉ là "chất liệu gợi ý" cho
teacher sinh (đề bài, lời giải) — KHÔNG phải dữ liệu train trực tiếp.

Mặc định quét `boltons` (BSD, đã cài cho M3). Thêm package khác qua --packages
để đạt số lượng (vd more-itertools MIT, toolz BSD — nhớ pip install trước).
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
from pathlib import Path
from typing import Any

MAX_SEED_LINES = 15
MIN_SEED_LINES = 1


def _license_hint(package: str) -> str:
    known = {
        "boltons": "BSD-3-Clause",
        "more_itertools": "MIT",
        "toolz": "BSD-3-Clause",
        "funcy": "BSD-3-Clause",
    }
    return known.get(package, "UNKNOWN — kiểm tra license trước khi dùng")


def collect_from_package(package: str) -> list[dict[str, Any]]:
    mod = importlib.import_module(package)
    root = Path(mod.__file__).resolve().parent
    license_hint = _license_hint(package)
    seeds: list[dict[str, Any]] = []

    for py_file in sorted(root.rglob("*.py")):
        source = py_file.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        rel = str(py_file.relative_to(root.parent))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name.startswith("_"):
                continue
            segment = ast.get_source_segment(source, node)
            if not segment:
                continue
            n_lines = len(segment.strip().splitlines())
            if not (MIN_SEED_LINES <= n_lines <= MAX_SEED_LINES):
                continue
            seeds.append(
                {
                    "seed_id": f"{package}:{node.name}:{node.lineno}",
                    "source_repo": package,
                    "license": license_hint,
                    "file": rel,
                    "lineno": node.lineno,
                    "snippet": segment.strip(),
                }
            )
    return seeds


def collect(packages: list[str]) -> list[dict[str, Any]]:
    seeds: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pkg in packages:
        for seed in collect_from_package(pkg):
            key = seed["snippet"]
            if key in seen:
                continue
            seen.add(key)
            seeds.append(seed)
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect open-license code seeds.")
    parser.add_argument("--packages", default="boltons", help="comma list of installed packages")
    parser.add_argument("--output", default="data/finetune/seeds.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="cắt còn N seed (0=tất cả)")
    args = parser.parse_args()

    packages = [p.strip() for p in args.packages.split(",") if p.strip()]
    seeds = collect(packages)
    if args.limit:
        seeds = seeds[: args.limit]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for seed in seeds:
            fh.write(json.dumps(seed, ensure_ascii=False) + "\n")

    by_repo: dict[str, int] = {}
    for seed in seeds:
        by_repo[seed["source_repo"]] = by_repo.get(seed["source_repo"], 0) + 1
    print(f"collected {len(seeds)} seeds -> {out}")
    for repo, count in by_repo.items():
        print(f"  {repo} ({_license_hint(repo)}): {count}")


if __name__ == "__main__":
    main()
