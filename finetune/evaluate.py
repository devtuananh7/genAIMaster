"""
Step 6 — Đánh giá model trước và sau fine-tune trên MBPP.

So sánh pass@1 của deepseek-coder-1.3b-base:
  - TRƯỚC fine-tune (base vanilla)
  - SAU fine-tune (base + LoRA adapter)

Chạy N=3 lần, báo mean ± std theo measurement_protocol trong 00-contract.yaml.

Usage:
    python -m finetune.evaluate --backend mlx
    python -m finetune.evaluate --backend transformers --adapter finetune/adapters/peft
    python -m finetune.evaluate --runs 3   # chạy 3 lần, tính mean±std
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from harness import executor
from harness.loader import load_tasks
from harness.scorer import iteration_payload, pass_at_1, write_results
from harness.types import ExecutionResult, Task

from finetune.strategy import FinetuneStrategy

# ── Paths ────────────────────────────────────────────────────────────────────
_MODULE_DIR = Path(__file__).resolve().parent
_DEFAULT_TASKS = Path("data/selected_tasks.json")
_RESULTS_DIR = _MODULE_DIR / "results"
_DEFAULT_ADAPTER_MLX = _MODULE_DIR / "adapters" / "mlx"
_DEFAULT_ADAPTER_PEFT = _MODULE_DIR / "adapters" / "peft"


def _run_evaluation(
    strategy: FinetuneStrategy,
    tasks: list[Task],
) -> tuple[float, list[dict[str, Any]]]:
    """Chạy 1 lần đánh giá, trả pass@1 và per-task results."""
    per_task: list[dict[str, Any]] = []

    for task in tasks:
        code = strategy.solve(task)
        result: ExecutionResult = executor.run(code, task)

        per_task.append({
            "task_id": task.task_id,
            "status": result.status,
            "iterations": [iteration_payload(1, code, result)],
            "error": {
                "status": result.status,
                "failed_test": result.failed_test,
                "traceback": result.traceback,
            } if result.status != "pass" else None,
        })

    results = [
        ExecutionResult(**item["iterations"][0]["execution"])
        for item in per_task
    ]
    score = pass_at_1(results)
    return score, per_task


def evaluate(
    *,
    backend: str = "mlx",
    model_path: str = "deepseek-ai/deepseek-coder-1.3b-base",
    adapter_path: str | None = None,
    tasks_path: Path = _DEFAULT_TASKS,
    runs: int = 3,
    prompt_style: str = "instruction",
) -> dict[str, Any]:
    """Đánh giá đầy đủ: trước/sau fine-tune, N runs."""
    tasks = load_tasks(tasks_path)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    results: dict[str, Any] = {
        "timestamp": timestamp,
        "model": model_path,
        "backend": backend,
        "adapter": adapter_path,
        "prompt_style": prompt_style,
        "n_tasks": len(tasks),
        "n_runs": runs,
        "before": {},
        "after": {},
    }

    # ═══════════════════════════════════════════════════════════════════════
    # TRƯỚC fine-tune (base vanilla)
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*60}", file=sys.stderr)
    print(f"  ĐÁNH GIÁ TRƯỚC FINE-TUNE (base vanilla)", file=sys.stderr)
    print(f"{'═'*60}", file=sys.stderr)

    before_strategy = FinetuneStrategy(
        backend=backend,
        model_path=model_path,
        adapter_path=None,  # KHÔNG load adapter
        prompt_style=prompt_style,
    )

    before_scores: list[float] = []
    before_all_per_task: list[list[dict]] = []

    for run_idx in range(1, runs + 1):
        print(f"\n  ── Run {run_idx}/{runs} (before) ──", file=sys.stderr)
        score, per_task = _run_evaluation(before_strategy, tasks)
        before_scores.append(score)
        before_all_per_task.append(per_task)
        print(f"  pass@1 = {score:.1%}", file=sys.stderr)

    results["before"] = {
        "scores": before_scores,
        "mean": statistics.mean(before_scores),
        "std": statistics.stdev(before_scores) if len(before_scores) > 1 else 0.0,
        "per_task_runs": before_all_per_task,
    }

    # ═══════════════════════════════════════════════════════════════════════
    # SAU fine-tune (base + LoRA)
    # ═══════════════════════════════════════════════════════════════════════
    if adapter_path and Path(adapter_path).exists():
        print(f"\n{'═'*60}", file=sys.stderr)
        print(f"  ĐÁNH GIÁ SAU FINE-TUNE (base + LoRA)", file=sys.stderr)
        print(f"  Adapter: {adapter_path}", file=sys.stderr)
        print(f"{'═'*60}", file=sys.stderr)

        after_strategy = FinetuneStrategy(
            backend=backend,
            model_path=model_path,
            adapter_path=adapter_path,
            prompt_style=prompt_style,
        )

        after_scores: list[float] = []
        after_all_per_task: list[list[dict]] = []

        for run_idx in range(1, runs + 1):
            print(f"\n  ── Run {run_idx}/{runs} (after) ──", file=sys.stderr)
            score, per_task = _run_evaluation(after_strategy, tasks)
            after_scores.append(score)
            after_all_per_task.append(per_task)
            print(f"  pass@1 = {score:.1%}", file=sys.stderr)

        results["after"] = {
            "scores": after_scores,
            "mean": statistics.mean(after_scores),
            "std": statistics.stdev(after_scores) if len(after_scores) > 1 else 0.0,
            "per_task_runs": after_all_per_task,
        }
    else:
        print(f"\n  ⚠ Adapter not found at '{adapter_path}' — skipping 'after' evaluation",
              file=sys.stderr)

    # ═══════════════════════════════════════════════════════════════════════
    # REPORT
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*60}", file=sys.stderr)
    print(f"  KẾT QUẢ SO SÁNH", file=sys.stderr)
    print(f"{'═'*60}", file=sys.stderr)

    before_mean = results["before"]["mean"]
    before_std = results["before"]["std"]
    print(f"  TRƯỚC fine-tune:  pass@1 = {before_mean:.1%} ± {before_std:.1%}", file=sys.stderr)

    if results["after"]:
        after_mean = results["after"]["mean"]
        after_std = results["after"]["std"]
        delta = after_mean - before_mean
        print(f"  SAU fine-tune:    pass@1 = {after_mean:.1%} ± {after_std:.1%}", file=sys.stderr)
        print(f"  Δ (improvement):  {delta:+.1%}", file=sys.stderr)
        results["delta"] = delta

    # Ghi kết quả
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = _RESULTS_DIR / f"eval_{timestamp}.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        f.write("\n")
    print(f"\n  Results saved to {output_file}", file=sys.stderr)

    return results


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Đánh giá trước/sau fine-tune trên MBPP (OSS-Instruct Step 6).",
    )
    parser.add_argument("--backend", default="mlx", choices=["mlx", "transformers"],
                        help="mlx = Mac | transformers = PC")
    parser.add_argument("--model", default="deepseek-ai/deepseek-coder-1.3b-base",
                        help="HuggingFace model ID")
    parser.add_argument("--adapter", default=None,
                        help="Path to LoRA adapter (default: auto-detect based on backend)")
    parser.add_argument("--tasks", default=str(_DEFAULT_TASKS),
                        help="Path to MBPP tasks JSON")
    parser.add_argument("--runs", type=int, default=3,
                        help="Số lần chạy lấy mean±std (default: 3)")
    parser.add_argument("--prompt-style", default="instruction",
                        choices=["instruction", "completion"],
                        help="Kiểu prompt (default: instruction)")
    args = parser.parse_args()

    # Auto-detect adapter path
    adapter = args.adapter
    if adapter is None:
        if args.backend == "mlx":
            candidate = _DEFAULT_ADAPTER_MLX
        else:
            candidate = _DEFAULT_ADAPTER_PEFT
        if candidate.exists():
            adapter = str(candidate)
            print(f"  Auto-detected adapter: {adapter}", file=sys.stderr)

    evaluate(
        backend=args.backend,
        model_path=args.model,
        adapter_path=adapter,
        tasks_path=Path(args.tasks),
        runs=args.runs,
        prompt_style=args.prompt_style,
    )


if __name__ == "__main__":
    main()
