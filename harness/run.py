from __future__ import annotations

import argparse
from typing import Any

from baseline.strategy import BaselineStrategy
from harness import executor
from harness.loader import load_tasks
from harness.ollama_client import DEFAULT_MODEL, resolve_model
from harness.scorer import iteration_payload, write_results
from harness.types import Strategy


def load_strategy(
    name: str,
    *,
    temperature: float,
    max_tokens: int,
    base_url: str | None,
    model: str | None,
) -> Strategy:
    if name == "baseline":
        return BaselineStrategy(
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            model=model,
        )
    raise ValueError(f"Unknown strategy: {name}")


def run_strategy(
    *,
    strategy: Strategy,
    tasks_path: str,
    temperature: float,
    max_tokens: int,
    samples: int,
    model: str | None,
) -> str:
    tasks = load_tasks(tasks_path)
    per_task: list[dict[str, Any]] = []

    print("task_id,status")
    for task in tasks:
        iterations: list[dict[str, Any]] = []
        for sample_index in range(1, samples + 1):
            code = strategy.solve(task)
            result = executor.run(code, task)
            iterations.append(iteration_payload(sample_index, code, result))
            if samples > 1 and result.status == "pass":
                break

        status = "pass" if any(
            item["execution"]["status"] == "pass" for item in iterations
        ) else iterations[0]["execution"]["status"]
        first_error = next(
            (item["execution"] for item in iterations if item["execution"]["status"] != "pass"),
            None,
        )
        per_task.append(
            {
                "task_id": task.task_id,
                "status": status,
                "iterations": iterations,
                "error": {
                    "status": first_error["status"],
                    "failed_test": first_error["failed_test"],
                    "traceback": first_error["traceback"],
                }
                if first_error
                else None,
            }
        )
        print(f"{task.task_id},{status}")

    output_path = write_results(
        strategy_name=strategy.name,
        model_name=resolve_model(model),
        params={
            "temperature": temperature,
            "max_tokens": max_tokens,
            "samples": samples,
        },
        tasks=tasks,
        per_task=per_task,
    )
    return str(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an MBPP strategy.")
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--base-url")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    strategy = load_strategy(
        args.strategy,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        base_url=args.base_url,
        model=args.model,
    )
    output_path = run_strategy(
        strategy=strategy,
        tasks_path=args.tasks,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        samples=args.samples,
        model=args.model,
    )
    print(f"results_path,{output_path}")


if __name__ == "__main__":
    main()
