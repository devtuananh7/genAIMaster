from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from harness.ollama_client import DEFAULT_MODEL
from harness.types import ExecutionResult, Task


def pass_at_1(results: list[ExecutionResult]) -> float:
    if not results:
        return 0.0
    return sum(result.status == "pass" for result in results) / len(results)


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

    first_results = [
        ExecutionResult(**item["iterations"][0]["execution"])
        for item in per_task
        if item.get("iterations")
    ]
    pass1 = pass_at_1(first_results)
    total_pass5 = None
    if params.get("samples", 1) >= 5:
        total_pass5 = sum(item["status"] == "pass" for item in per_task) / len(per_task)

    payload: dict[str, Any] = {
        "model_name": model_name,
        "params": params,
        "timestamp": timestamp,
        "task_list": [task.task_id for task in tasks],
        "per_task": per_task,
        "total_pass1": pass1,
    }
    if total_pass5 is not None:
        payload["total_pass5"] = total_pass5
        payload["run_type"] = "sampling"

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return output_path


def iteration_payload(sample_index: int, code: str, result: ExecutionResult) -> dict[str, Any]:
    return {
        "sample_index": sample_index,
        "code": code,
        "execution": asdict(result),
    }
