"""
Step 1 — Thu thập seed snippets từ các repo mã nguồn mở.

Quy trình OSS-Instruct: lấy 300-500 đoạn code ngắn (1-15 dòng body) từ repo
Python giấy phép mở, dùng làm "mã hạt giống" để giáo viên sinh cặp (đề, lời giải).

Usage:
    python -m finetune.collect_seeds
    python -m finetune.collect_seeds --target 400 --max-lines 12
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
import textwrap
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
_MODULE_DIR = Path(__file__).resolve().parent
_REPOS_DIR = _MODULE_DIR / "repos"
_DEFAULT_REPOS = _MODULE_DIR / "data" / "repos.json"
_DEFAULT_OUTPUT = _MODULE_DIR / "data" / "seed_snippets.json"

# ── Tiêu chí lọc ────────────────────────────────────────────────────────────
MIN_BODY_LINES = 1
MAX_BODY_LINES = 15
SEED = 5410  # reproducible sampling

# Hàm trivial — bỏ qua
_TRIVIAL_NAMES = frozenset({
    "__init__", "__repr__", "__str__", "__len__", "__eq__", "__hash__",
    "__lt__", "__le__", "__gt__", "__ge__", "__ne__", "__bool__",
    "__enter__", "__exit__", "__del__", "__new__",
    "setUp", "tearDown", "setUpClass", "tearDownClass",
    "main",
})

# Module bên thứ ba phổ biến — lọc ra
_THIRD_PARTY = frozenset({
    "numpy", "np", "pandas", "pd", "torch", "tensorflow", "tf",
    "sklearn", "scipy", "matplotlib", "plt", "cv2", "PIL",
    "flask", "django", "fastapi", "requests", "httpx", "aiohttp",
    "sqlalchemy", "pydantic", "pytest", "setuptools", "pip",
    "click", "typer", "rich", "tqdm",
})

# Thư mục bỏ qua khi quét
_SKIP_DIRS = frozenset({
    "test", "tests", "testing", "docs", "doc", "examples", "example",
    "venv", ".venv", "env", "__pycache__", ".git", ".github",
    "node_modules", "build", "dist", "egg-info", ".tox", ".nox",
    "scripts", "benchmarks", "benchmark", "setup",
})


# ── Helpers ──────────────────────────────────────────────────────────────────

def _clone_repos(repos_file: Path) -> list[dict]:
    """Clone (shallow) các repo từ danh sách."""
    with repos_file.open("r", encoding="utf-8") as f:
        repos = json.load(f)

    _REPOS_DIR.mkdir(parents=True, exist_ok=True)

    for repo in repos:
        local_name = repo["name"].replace("/", "__")
        dest = _REPOS_DIR / local_name
        if dest.exists():
            print(f"  [skip] {repo['name']} — already at {dest}", file=sys.stderr)
            continue
        print(f"  [clone] {repo['name']} → {dest}", file=sys.stderr)
        subprocess.run(
            ["git", "clone", "--depth", "1", repo["url"], str(dest)],
            check=True,
            capture_output=True,
            text=True,
        )

    return repos


def _body_lines(node: ast.FunctionDef, source_lines: list[str]) -> list[str]:
    """Trả về các dòng body (bỏ docstring, bỏ dòng trống)."""
    body = node.body
    # Bỏ docstring nếu có
    if (body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, (ast.Constant, ast.Str))):
        body = body[1:]

    if not body:
        return []

    start = body[0].lineno - 1  # 0-indexed
    end = node.end_lineno or start + 1
    raw = source_lines[start:end]
    return [ln for ln in raw if ln.strip()]


def _full_source(node: ast.FunctionDef, source_lines: list[str]) -> str:
    """Trích toàn bộ mã nguồn hàm."""
    start = node.lineno - 1
    end = node.end_lineno or start + 1
    lines = source_lines[start:end]
    return textwrap.dedent("\n".join(lines)).strip()


def _is_trivial(node: ast.FunctionDef) -> bool:
    """Hàm quá đơn giản — không có giá trị làm seed."""
    name = node.name
    if name in _TRIVIAL_NAMES:
        return True
    if name.startswith("test_") or name.startswith("_test"):
        return True

    body = node.body
    # Bỏ docstring
    if (body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, (ast.Constant, ast.Str))):
        body = body[1:]

    if not body:
        return True
    # Chỉ có 1 statement: pass / return None / raise
    if len(body) == 1:
        stmt = body[0]
        if isinstance(stmt, ast.Pass):
            return True
        if isinstance(stmt, ast.Return) and stmt.value is None:
            return True
        if isinstance(stmt, ast.Raise) and stmt.exc is None:
            return True
    return False


def _touches_third_party(node: ast.FunctionDef) -> bool:
    """Heuristic: hàm có tham chiếu thư viện bên thứ ba?"""
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id in _THIRD_PARTY:
            return True
        if isinstance(child, ast.Attribute):
            if isinstance(child.value, ast.Name) and child.value.id in _THIRD_PARTY:
                return True
    return False


def _extract_from_file(
    py_file: Path,
    repo_name: str,
    source_lines: list[str],
    tree: ast.Module,
    *,
    min_lines: int,
    max_lines: int,
) -> list[dict]:
    """Trích xuất các hàm hợp lệ từ 1 file Python đã parse."""
    snippets: list[dict] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if _is_trivial(node):
            continue
        if _touches_third_party(node):
            continue

        body = _body_lines(node, source_lines)
        n = len(body)
        if n < min_lines or n > max_lines:
            continue

        code = _full_source(node, source_lines)
        if not code:
            continue

        snippets.append({
            "repo": repo_name,
            "file": str(py_file),
            "function_name": node.name,
            "start_line": node.lineno,
            "end_line": node.end_lineno,
            "body_line_count": n,
            "code": code,
        })

    return snippets


# ── Public API ───────────────────────────────────────────────────────────────

def collect(
    repos_file: Path = _DEFAULT_REPOS,
    output: Path = _DEFAULT_OUTPUT,
    *,
    min_lines: int = MIN_BODY_LINES,
    max_lines: int = MAX_BODY_LINES,
    target_count: int = 500,
) -> list[dict]:
    """Pipeline chính: clone → trích hàm → lọc → ghi file."""
    # ── Clone ────────────────────────────────────────────────────────────
    print("═══ Step 1/3: Clone repos ═══", file=sys.stderr)
    repos = _clone_repos(repos_file)

    # ── Extract ──────────────────────────────────────────────────────────
    print("═══ Step 2/3: Extract functions ═══", file=sys.stderr)
    all_snippets: list[dict] = []

    for repo_info in repos:
        name = repo_info["name"]
        local = _REPOS_DIR / name.replace("/", "__")
        if not local.exists():
            print(f"  [warn] {name} — not found, skipping", file=sys.stderr)
            continue

        py_files = sorted(local.rglob("*.py"))
        repo_snippets: list[dict] = []

        for py_file in py_files:
            # Bỏ thư mục không phù hợp
            rel = py_file.relative_to(local)
            if any(part.lower() in _SKIP_DIRS or part.endswith(".egg-info")
                   for part in rel.parts):
                continue

            try:
                source = py_file.read_text(encoding="utf-8", errors="ignore")
                tree = ast.parse(source)
            except (SyntaxError, UnicodeDecodeError, ValueError):
                continue

            source_lines = source.splitlines()
            funcs = _extract_from_file(
                rel, name, source_lines, tree,
                min_lines=min_lines, max_lines=max_lines,
            )
            repo_snippets.extend(funcs)

        # Đường dẫn tương đối cho đẹp
        for s in repo_snippets:
            s["file"] = str(s["file"])

        print(f"  {name}: {len(repo_snippets)} hàm hợp lệ", file=sys.stderr)
        all_snippets.extend(repo_snippets)

    # ── Sample ───────────────────────────────────────────────────────────
    print(f"\n═══ Step 3/3: Sample & save ═══", file=sys.stderr)
    print(f"  Tổng cộng: {len(all_snippets)} snippets", file=sys.stderr)

    if len(all_snippets) > target_count:
        import random
        rng = random.Random(SEED)
        all_snippets = rng.sample(all_snippets, target_count)
        print(f"  Lấy mẫu xuống {target_count}", file=sys.stderr)
    elif len(all_snippets) < target_count:
        print(
            f"  ⚠  Chỉ thu được {len(all_snippets)}/{target_count} — "
            f"cân nhắc thêm repo hoặc nới tiêu chí lọc",
            file=sys.stderr,
        )

    # Ghi file
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(all_snippets, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"  ✓ Saved {len(all_snippets)} seeds → {output}", file=sys.stderr)
    return all_snippets


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Thu thập seed snippets từ repo mã nguồn mở (OSS-Instruct Step 1).",
    )
    parser.add_argument("--repos", default=str(_DEFAULT_REPOS),
                        help="File JSON danh sách repo (default: data/repos.json)")
    parser.add_argument("--output", default=str(_DEFAULT_OUTPUT),
                        help="File output (default: data/seed_snippets.json)")
    parser.add_argument("--min-lines", type=int, default=MIN_BODY_LINES,
                        help="Số dòng body tối thiểu (default: 1)")
    parser.add_argument("--max-lines", type=int, default=MAX_BODY_LINES,
                        help="Số dòng body tối đa (default: 15)")
    parser.add_argument("--target", type=int, default=500,
                        help="Số snippet mục tiêu (default: 500)")
    args = parser.parse_args()

    collect(
        repos_file=Path(args.repos),
        output=Path(args.output),
        min_lines=args.min_lines,
        max_lines=args.max_lines,
        target_count=args.target,
    )


if __name__ == "__main__":
    main()
