"""
finetune/filter_runnable.py  — Stage A, bước 2 (lọc)
====================================================
Loại các mẫu teacher mà lời giải KHÔNG tự chạy được — bằng cách TÁI DÙNG
Sandbox Executor của M0 (điểm cộng hệ thống theo spec M5).

Mỗi mẫu {solution, tests} được chạy như một Task M0; giữ lại nếu status == "pass".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from harness import executor
from harness.types import Task


def is_runnable(sample: dict[str, Any]) -> tuple[bool, str]:
    task = Task(
        task_id=0,
        text=sample.get("problem", ""),
        test_list=list(sample.get("tests", [])),
        test_imports=[],
    )
    result = executor.run(sample["solution"], task)
    return result.status == "pass", result.status


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter teacher samples by executability (M0).")
    parser.add_argument("--input", default="data/finetune/teacher_raw.jsonl")
    parser.add_argument("--output", default="data/finetune/teacher_filtered.jsonl")
    args = parser.parse_args()

    samples = [json.loads(l) for l in Path(args.input).read_text().splitlines() if l.strip()]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    reasons: dict[str, int] = {}
    with out.open("w", encoding="utf-8") as fh:
        for sample in samples:
            ok, status = is_runnable(sample)
            reasons[status] = reasons.get(status, 0) + 1
            if ok:
                fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
                kept += 1
    print(f"kept {kept}/{len(samples)} runnable samples -> {out}")
    print(f"  status breakdown: {reasons}")


if __name__ == "__main__":
    main()
