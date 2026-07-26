from __future__ import annotations

import argparse
from typing import Any

from baseline.strategy import BaselineStrategy
from harness import executor
from harness.loader import load_tasks
from harness.hf_client import DEFAULT_HF_MODEL as DEFAULT_MODEL, resolve_model
from harness.scorer import write_results
from harness.types import Strategy
from reflexion.strategy import ReflexionStrategy
from multiagent.strategy import MultiAgentStrategy


def load_strategy(
    name: str,
    *,
    temperature: float,
    max_tokens: int,
    base_url: str | None,
    model: str | None,
    max_iterations: int | None = None,
) -> Strategy:
    if name == "baseline":
        return BaselineStrategy(
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            model=model,
        )
    elif name == "reflexion":
        kwargs: dict = dict(
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            model=model,
        )
        if max_iterations is not None:
            kwargs["max_iterations"] = max_iterations
        return ReflexionStrategy(**kwargs)
    elif name == "multiagent":
        kwargs_ma: dict = dict(
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            model=model,
        )
        if max_iterations is not None:
            kwargs_ma["max_iterations"] = max_iterations
        return MultiAgentStrategy(**kwargs_ma)
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
        code = strategy.solve(task)
        result = executor.run(code, task)

        # Đọc metadata từ strategy (được gán trong solve())
        rounds_to_pass = getattr(strategy, "_rounds_to_pass", None)
        total_rounds = getattr(strategy, "_total_rounds", 1)
        pass_1st_round = getattr(strategy, "_pass_1st_round", False)
        internal_records = getattr(strategy, "_internal_records", [])
        final_code = getattr(strategy, "_final_code", code)

        # Baseline: không execute nội bộ → điền execution từ run.py
        if strategy.name == "baseline":
            if result.status == "pass":
                rounds_to_pass = 1
                pass_1st_round = True
            # Điền execution vào internal_records (baseline chỉ có 1 record)
            if internal_records and internal_records[0].get("execution") is None:
                internal_records[0]["execution"] = {
                    "status": result.status,
                    "failed_test": result.failed_test,
                    "traceback": result.traceback,
                    "passed_count": result.passed_count,
                    "total_count": result.total_count,
                    "duration_ms": result.duration_ms,
                }

        status = result.status if strategy.name == "baseline" else (
            "pass" if rounds_to_pass is not None else (
                internal_records[-1]["execution"]["status"]
                if internal_records else result.status
            )
        )

        per_task.append(
            {
                "task_id": task.task_id,
                "status": status,
                "rounds_to_pass": rounds_to_pass,
                "total_rounds": total_rounds,
                "pass_1st_round": pass_1st_round,
                "final_code": final_code,
                "internal_records": internal_records,
            }
        )
        print(f"{task.task_id},{status}")

    # Lấy max_iterations và seed từ strategy (nếu có)
    max_iterations = getattr(strategy, "max_iterations", 1)
    from harness.hf_client import DEFAULT_SEED
    seed = DEFAULT_SEED

    output_path = write_results(
        strategy_name=strategy.name,
        model_name=resolve_model(model),
        params={
            "temperature": temperature,
            "max_tokens": max_tokens,
            "max_iterations": max_iterations,
            "seed": seed,
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
    parser.add_argument("--max-iterations", type=int, default=None,
                        help="Max loop iterations for reflexion/multiagent")
    args = parser.parse_args()

    strategy = load_strategy(
        args.strategy,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        base_url=args.base_url,
        model=args.model,
        max_iterations=args.max_iterations,
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
