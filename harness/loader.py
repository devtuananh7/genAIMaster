from __future__ import annotations

import argparse
import ast
import json
import random
import sys
from pathlib import Path
from typing import Any

from harness.types import Task

SUBSET_SIZE = 50
QUINTILE_COUNT = 5
TASKS_PER_QUINTILE = 10
SEED = 5410


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _stdlib_roots(import_lines: list[str]) -> set[str]:
    roots: set[str] = set()
    for line in import_lines:
        try:
            module = ast.parse(line)
        except SyntaxError:
            return {"<invalid>"}
        for node in module.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    roots.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    roots.add(node.module.split(".")[0])
    return roots


def _uses_only_stdlib(record: dict[str, Any]) -> bool:
    roots = _stdlib_roots([str(line) for line in _as_list(record.get("test_imports"))])
    stdlib = getattr(sys, "stdlib_module_names", set())
    return all(root in stdlib for root in roots)


def _reference_line_count(record: dict[str, Any]) -> int:
    code = str(record.get("code", ""))
    return max(1, len(code.strip().splitlines()))


def _task_text(record: dict[str, Any]) -> str:
    return str(record.get("text") or record.get("prompt") or "")


def _normalize_task(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": int(record["task_id"]),
        "text": _task_text(record),
        "test_list": [str(test) for test in _as_list(record["test_list"])],
        "test_imports": [str(line) for line in _as_list(record.get("test_imports"))],
    }


def _all_records(dataset: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for split in dataset:
        records.extend(dict(row) for row in dataset[split])
    return records


def select_tasks(seed: int = SEED, size: int = SUBSET_SIZE) -> list[dict[str, Any]]:
    from datasets import load_dataset

    if size % QUINTILE_COUNT != 0:
        raise ValueError(f"size ({size}) phải chia hết cho {QUINTILE_COUNT} quintile")
    per_quintile = size // QUINTILE_COUNT

    try:
        dataset = load_dataset("mbpp", "sanitized")
    except Exception as exc:
        message = str(exc)
        if (
            "Repository id must be 'namespace/name'" not in message
            and "Dataset scripts are no longer supported" not in message
        ):
            raise
        dataset = load_dataset("claudios/google-research-datasets__mbpp", "sanitized")
    candidates = [record for record in _all_records(dataset) if _uses_only_stdlib(record)]
    candidates.sort(key=lambda record: (_reference_line_count(record), int(record["task_id"])))

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for idx in range(QUINTILE_COUNT):
        start = idx * len(candidates) // QUINTILE_COUNT
        end = (idx + 1) * len(candidates) // QUINTILE_COUNT
        bucket = candidates[start:end]
        if len(bucket) < per_quintile:
            raise RuntimeError(f"Not enough MBPP candidates in bucket {idx + 1} (cần {per_quintile}, có {len(bucket)})")
        selected.extend(rng.sample(bucket, per_quintile))

    selected.sort(key=lambda record: int(record["task_id"]))
    if len(selected) != size:
        raise RuntimeError(f"Expected {size} tasks, selected {len(selected)}")
    return [_normalize_task(record) for record in selected]


def write_selected_tasks(path: str | Path, *, overwrite: bool = False, size: int = SUBSET_SIZE) -> list[dict[str, Any]]:
    output_path = Path(path)
    if output_path.exists() and not overwrite:
        with output_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    tasks = select_tasks(size=size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(tasks, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return tasks


def load_tasks(path: str | Path) -> list[Task]:
    with Path(path).open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return [
        Task(
            task_id=int(item["task_id"]),
            text=str(item["text"]),
            test_list=[str(test) for test in item["test_list"]],
            test_imports=[str(line) for line in item.get("test_imports", [])],
        )
        for item in payload
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Select and freeze 50 MBPP tasks.")
    parser.add_argument("--output", default="data/selected_tasks.json")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--size", type=int, default=SUBSET_SIZE)
    args = parser.parse_args()

    tasks = write_selected_tasks(args.output, overwrite=args.overwrite, size=args.size)
    print(f"wrote {len(tasks)} tasks to {args.output}")


if __name__ == "__main__":
    main()
