"""
Step 3 — Lọc dữ liệu qua Sandbox Executor (M0).

Chạy lời giải sinh bởi giáo viên qua bộ kiểm thử để loại bỏ mẫu không hợp lệ.
Tái sử dụng harness.executor.run() — điểm cộng hệ thống (reuse M0).

Usage:
    python -m finetune.filter_data
    python -m finetune.filter_data --input finetune/data/raw_pairs.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from harness.executor import run as sandbox_run
from harness.types import ExecutionResult, Task

# ── Paths ────────────────────────────────────────────────────────────────────
_MODULE_DIR = Path(__file__).resolve().parent
_DEFAULT_INPUT = _MODULE_DIR / "data" / "raw_pairs.json"
_DEFAULT_OUTPUT = _MODULE_DIR / "data" / "filtered_pairs.json"
_DEFAULT_STATS = _MODULE_DIR / "data" / "filter_stats.json"


def _pair_to_task(pair: dict) -> Task:
    """Tạo Task tạm từ cặp dữ liệu teacher để chạy qua executor."""
    return Task(
        task_id=-1,  # không thuộc MBPP
        text=pair["problem"],
        test_list=pair["tests"],
        test_imports=[],
    )


def filter_pairs(
    input_file: Path = _DEFAULT_INPUT,
    output_file: Path = _DEFAULT_OUTPUT,
    stats_file: Path = _DEFAULT_STATS,
) -> list[dict]:
    """Lọc raw pairs: chỉ giữ mẫu có lời giải chạy pass toàn bộ test."""
    with input_file.open("r", encoding="utf-8") as f:
        raw_pairs = json.load(f)

    print(f"═══ Filtering {len(raw_pairs)} raw pairs ═══", file=sys.stderr)

    filtered: list[dict] = []
    stats = {
        "total": len(raw_pairs),
        "pass": 0,
        "fail_assert": 0,
        "error_syntax": 0,
        "error_runtime": 0,
        "timeout": 0,
    }

    for idx, pair in enumerate(raw_pairs):
        progress = f"[{idx+1}/{len(raw_pairs)}]"
        entry = pair.get("entry_point", "?")

        # Tạo task tạm
        task = _pair_to_task(pair)
        code = pair["solution"]

        # Chạy qua sandbox executor (reuse M0)
        try:
            result: ExecutionResult = sandbox_run(code, task)
        except Exception as exc:
            print(f"  {progress} {entry} — ✗ executor error: {exc}", file=sys.stderr)
            stats["error_runtime"] += 1
            continue

        status = result.status
        stats[status] = stats.get(status, 0) + 1

        if status == "pass":
            pair_with_result = {
                **pair,
                "filter_status": "pass",
                "passed_count": result.passed_count,
                "total_count": result.total_count,
            }
            filtered.append(pair_with_result)
            print(f"  {progress} {entry} — ✓ pass ({result.passed_count}/{result.total_count})",
                  file=sys.stderr)
        else:
            detail = result.traceback[:80] if result.traceback else ""
            print(f"  {progress} {entry} — ✗ {status}: {detail}", file=sys.stderr)

    # Ghi kết quả
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)
        f.write("\n")

    with stats_file.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
        f.write("\n")

    # Report
    pass_rate = stats["pass"] / stats["total"] * 100 if stats["total"] > 0 else 0
    print(f"\n═══ Filter Results ═══", file=sys.stderr)
    print(f"  Total:        {stats['total']}", file=sys.stderr)
    print(f"  ✓ Pass:       {stats['pass']} ({pass_rate:.1f}%)", file=sys.stderr)
    print(f"  ✗ Fail assert:{stats.get('fail_assert', 0)}", file=sys.stderr)
    print(f"  ✗ Syntax err: {stats.get('error_syntax', 0)}", file=sys.stderr)
    print(f"  ✗ Runtime err:{stats.get('error_runtime', 0)}", file=sys.stderr)
    print(f"  ✗ Timeout:    {stats.get('timeout', 0)}", file=sys.stderr)
    print(f"  Output:       {output_file}", file=sys.stderr)

    return filtered


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lọc dữ liệu qua Sandbox Executor (OSS-Instruct Step 3).",
    )
    parser.add_argument("--input", default=str(_DEFAULT_INPUT),
                        help="File raw pairs (default: data/raw_pairs.json)")
    parser.add_argument("--output", default=str(_DEFAULT_OUTPUT),
                        help="File output (default: data/filtered_pairs.json)")
    args = parser.parse_args()

    filter_pairs(
        input_file=Path(args.input),
        output_file=Path(args.output),
    )


if __name__ == "__main__":
    main()
