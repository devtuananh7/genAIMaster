"""
harness/scorer.py
=================
Tính toán metric, xuất kết quả JSON + CSV cho mỗi lần chạy,
và tổng hợp (append) vào file summary JSON/CSV.
"""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from harness.ollama_client import DEFAULT_MODEL
from harness.types import Task


# ---------------------------------------------------------------------------
# Tính pass@1
# ---------------------------------------------------------------------------
def pass_at_1(per_task: list[dict[str, Any]]) -> float:
    """Tính pass@1 từ danh sách per_task."""
    if not per_task:
        return 0.0
    return sum(item["status"] == "pass" for item in per_task) / len(per_task)


# ---------------------------------------------------------------------------
# Xuất CSV cho 1 lần chạy (per-run CSV)
# ---------------------------------------------------------------------------
def write_csv(
    *,
    output_path: Path,
    strategy_name: str,
    model_name: str,
    per_task: list[dict[str, Any]],
    pass1: float,
) -> Path:
    """
    Xuất bảng tổng hợp CSV cùng thư mục với file JSON.
    Mỗi dòng = 1 task, cuối file có dòng tổng hợp pass@1.
    """
    csv_path = output_path.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "strategy",
            "model",
            "task_id",
            "status",
            "pass_1st_round",
            "rounds_to_pass",
            "total_rounds",
            "failed_test",
        ])
        for item in per_task:
            # Lấy failed_test từ record cuối cùng (trạng thái cuối)
            failed_test = _get_failed_test(item)
            writer.writerow([
                strategy_name,
                model_name,
                item["task_id"],
                item["status"],
                item.get("pass_1st_round", ""),
                item.get("rounds_to_pass", ""),
                item.get("total_rounds", ""),
                failed_test,
            ])
        # Dòng tổng hợp
        writer.writerow([])
        writer.writerow(["strategy", strategy_name])
        writer.writerow(["model", model_name])
        writer.writerow(["total_tasks", len(per_task)])
        writer.writerow(["pass@1", f"{pass1:.4f}"])
    return csv_path


def _get_failed_test(item: dict[str, Any]) -> str:
    """Trích failed_test từ internal_records (record cuối cùng nếu fail)."""
    records = item.get("internal_records", [])
    if not records:
        return ""
    last = records[-1]
    exec_data = last.get("execution", {}) or {}
    return exec_data.get("failed_test", "") or ""


# ---------------------------------------------------------------------------
# Tổng hợp summary JSON + CSV (append mỗi lần chạy mới)
# ---------------------------------------------------------------------------
def append_to_summary(
    *,
    output_dir: Path,
    strategy_name: str,
    model_name: str,
    timestamp: str,
    pass1: float,
    per_task: list[dict[str, Any]],
) -> tuple[Path, Path]:
    """
    Thêm bản ghi kết quả vào file tổng hợp summary.json và summary.csv.
    Nếu file chưa tồn tại sẽ tạo mới. Nếu đã tồn tại sẽ append.
    """
    # --- Summary JSON: mảng các bản ghi run ---
    summary_json_path = output_dir / "summary.json"
    run_summary = {
        "strategy": strategy_name,
        "model": model_name,
        "timestamp": timestamp,
        "total_pass1": pass1,
        "total_tasks": len(per_task),
        "per_task_summary": [
            {
                "task_id": item["task_id"],
                "status": item["status"],
                "rounds_to_pass": item.get("rounds_to_pass"),
                "total_rounds": item.get("total_rounds"),
                "pass_1st_round": item.get("pass_1st_round"),
            }
            for item in per_task
        ],
    }
    # Đọc file cũ (nếu có) rồi append
    runs: list[dict[str, Any]] = []
    if summary_json_path.exists():
        try:
            with summary_json_path.open("r", encoding="utf-8") as fh:
                runs = json.load(fh)
        except (json.JSONDecodeError, OSError):
            runs = []
    runs.append(run_summary)
    with summary_json_path.open("w", encoding="utf-8") as fh:
        json.dump(runs, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    # --- Summary CSV: append dòng mới ---
    summary_csv_path = output_dir / "summary.csv"
    csv_exists = summary_csv_path.exists()
    with summary_csv_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        # Viết header nếu file mới
        if not csv_exists:
            writer.writerow([
                "strategy",
                "model",
                "timestamp",
                "task_id",
                "status",
                "pass_1st_round",
                "rounds_to_pass",
                "total_rounds",
                "failed_test",
            ])
        for item in per_task:
            failed_test = _get_failed_test(item)
            writer.writerow([
                strategy_name,
                model_name,
                timestamp,
                item["task_id"],
                item["status"],
                item.get("pass_1st_round", ""),
                item.get("rounds_to_pass", ""),
                item.get("total_rounds", ""),
                failed_test,
            ])

    return summary_json_path, summary_csv_path


# ---------------------------------------------------------------------------
# Hàm chính: xuất kết quả (per-run JSON + CSV + tổng hợp)
# ---------------------------------------------------------------------------
def write_results(
    *,
    strategy_name: str,
    model_name: str = DEFAULT_MODEL,
    params: dict[str, Any],
    tasks: list[Task],
    per_task: list[dict[str, Any]],
) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path("results") / strategy_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{timestamp}.json"

    pass1 = pass_at_1(per_task)

    # --- Per-run JSON (theo schema mới) ---
    payload: dict[str, Any] = {
        "strategy": strategy_name,
        "model_name": model_name,
        "timestamp": timestamp,
        "params": params,
        "task_list": [task.task_id for task in tasks],
        "total_pass1": pass1,
        "per_task": per_task,
    }

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"json_path,{output_path}")

    # --- Per-run CSV ---
    csv_path = write_csv(
        output_path=output_path,
        strategy_name=strategy_name,
        model_name=model_name,
        per_task=per_task,
        pass1=pass1,
    )
    print(f"csv_path,{csv_path}")

    # --- Tổng hợp summary (append) ---
    summary_json, summary_csv = append_to_summary(
        output_dir=output_dir,
        strategy_name=strategy_name,
        model_name=model_name,
        timestamp=timestamp,
        pass1=pass1,
        per_task=per_task,
    )
    print(f"summary_json,{summary_json}")
    print(f"summary_csv,{summary_csv}")

    return output_path
