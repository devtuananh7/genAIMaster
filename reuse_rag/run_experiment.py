"""
reuse_rag/run_experiment.py
===========================
Driver thí nghiệm REPRESENTATION: quét 4 mức context × N task, đo mỗi mức:

  - reuse_rate : tỉ lệ code sinh ra TÁI DÙNG đúng API đích   (metric headline)
  - pass_rate  : tỉ lệ code chạy qua unit test               (lan can correctness)
  - avg_tokens : token context trung bình                     (trục chi phí)

Xuất: results/reuse_rag/<ts>.json (chi tiết), curve_<ts>.csv (đường cong),
và in bảng + đường cong ASCII ra stdout.

Chạy thật (cần HF_TOKEN):
    export HF_TOKEN=...
    ./.venv/bin/python -m reuse_rag.run_experiment --samples 3

Tự kiểm thử plumbing (không cần token):
    ./.venv/bin/python -m reuse_rag.run_experiment --mock
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from harness import executor
from harness.hf_client import resolve_model
from reuse_rag.indexer import load_index, write_index
from reuse_rag.render import LEVEL_NAMES, LEVELS
from reuse_rag.reuse_scorer import score_reuse
from reuse_rag.strategy import ReuseRagStrategy, make_mock_generate
from reuse_rag.tasks import load_repo_tasks

RESULTS_DIR = Path("results") / "reuse_rag"
DEFAULT_TASKS = "data/repo_tasks.json"
DEFAULT_INDEX = "data/boltons_index.json"


def ensure_index(index_path: str, package: str) -> None:
    if not Path(index_path).exists():
        print(f"index missing → building from package {package!r} ...")
        idx = write_index(index_path, package=package)
        print(f"  indexed {len(idx)} symbols → {index_path}")


def run_level(
    *,
    level: int,
    tasks: list,
    samples: int,
    temperature: float,
    max_tokens: int,
    index_path: str,
    base_url: str | None,
    model: str | None,
    generate_fn,
) -> dict[str, Any]:
    strategy = ReuseRagStrategy(
        level=level,
        index_path=index_path,
        temperature=temperature,
        max_tokens=max_tokens,
        base_url=base_url,
        model=model,
        generate_fn=generate_fn,
    )

    per_task: list[dict[str, Any]] = []
    for task in tasks:
        reuse_hits = 0
        pass_hits = 0
        context_tokens = 0
        samples_detail: list[dict[str, Any]] = []
        for _ in range(samples):
            code = strategy.solve(task)
            context_tokens = strategy._context_tokens
            reuse = score_reuse(code, task.target_module, task.target_name)
            exec_result = executor.run(code, task.to_harness_task())
            reuse_hits += int(reuse.reused)
            pass_hits += int(exec_result.status == "pass")
            samples_detail.append(
                {
                    "reused": reuse.reused,
                    "reuse_detail": reuse.detail,
                    "self_defined": reuse.self_defined,
                    "status": exec_result.status,
                    "code": code,
                }
            )
        per_task.append(
            {
                "task_id": task.task_id,
                "wrapper": task.wrapper,
                "target_fqn": task.target_fqn,
                "context_tokens": context_tokens,
                "reuse_rate": reuse_hits / samples,
                "pass_rate": pass_hits / samples,
                "samples": samples_detail,
            }
        )

    n = len(per_task) or 1
    return {
        "level": level,
        "level_name": LEVEL_NAMES[level],
        "reuse_rate": sum(t["reuse_rate"] for t in per_task) / n,
        "pass_rate": sum(t["pass_rate"] for t in per_task) / n,
        "avg_context_tokens": sum(t["context_tokens"] for t in per_task) / n,
        "per_task": per_task,
    }


def ascii_curve(levels_summary: list[dict[str, Any]]) -> str:
    """Vẽ đường cong reuse-rate theo mức context bằng ASCII."""
    lines = ["", "reuse-rate theo mức context (▇ = 5%):", ""]
    for row in levels_summary:
        bars = "▇" * round(row["reuse_rate"] * 20)
        lines.append(
            f"  {row['level_name']:<14} "
            f"reuse={row['reuse_rate']*100:5.1f}%  "
            f"pass={row['pass_rate']*100:5.1f}%  "
            f"~tok={row['avg_context_tokens']:5.1f}  |{bars}"
        )
    lines.append("")
    return "\n".join(lines)


def write_outputs(
    *,
    levels_summary: list[dict[str, Any]],
    params: dict[str, Any],
    model_name: str,
    mock: bool,
) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    tag = "MOCK_" if mock else ""

    json_path = RESULTS_DIR / f"{tag}{ts}.json"
    payload = {
        "experiment": "input_representation_reuse",
        "mock": mock,
        "model_name": model_name,
        "timestamp": ts,
        "params": params,
        "curve": [
            {
                "level": r["level"],
                "level_name": r["level_name"],
                "reuse_rate": r["reuse_rate"],
                "pass_rate": r["pass_rate"],
                "avg_context_tokens": r["avg_context_tokens"],
            }
            for r in levels_summary
        ],
        "levels": levels_summary,
    }
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    curve_path = RESULTS_DIR / f"{tag}curve_{ts}.csv"
    with curve_path.open("w", encoding="utf-8") as fh:
        fh.write("level,level_name,reuse_rate,pass_rate,avg_context_tokens\n")
        for r in levels_summary:
            fh.write(
                f"{r['level']},{r['level_name']},{r['reuse_rate']:.4f},"
                f"{r['pass_rate']:.4f},{r['avg_context_tokens']:.2f}\n"
            )
    return json_path, curve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Input-representation reuse experiment.")
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--package", default="boltons")
    parser.add_argument("--levels", default="1,2,3,4", help="comma list of levels")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=0, help="limit number of tasks (0=all)")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--mock", action="store_true", help="use offline mock generator")
    args = parser.parse_args()

    ensure_index(args.index, args.package)
    tasks = load_repo_tasks(args.tasks)
    if args.limit:
        tasks = tasks[: args.limit]
    levels = [int(x) for x in args.levels.split(",") if x.strip()]

    generate_fn = make_mock_generate() if args.mock else None

    print(f"tasks={len(tasks)}  levels={levels}  samples={args.samples}  mock={args.mock}")
    levels_summary: list[dict[str, Any]] = []
    for level in levels:
        summary = run_level(
            level=level,
            tasks=tasks,
            samples=args.samples,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            index_path=args.index,
            base_url=args.base_url,
            model=args.model,
            generate_fn=generate_fn,
        )
        levels_summary.append(summary)
        print(
            f"  {summary['level_name']:<14} "
            f"reuse={summary['reuse_rate']*100:5.1f}%  "
            f"pass={summary['pass_rate']*100:5.1f}%  "
            f"~tok={summary['avg_context_tokens']:.1f}"
        )

    params = {
        "samples": args.samples,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "levels": levels,
        "package": args.package,
        "n_tasks": len(tasks),
    }
    model_name = "MOCK" if args.mock else resolve_model(args.model)
    json_path, curve_path = write_outputs(
        levels_summary=levels_summary,
        params=params,
        model_name=model_name,
        mock=args.mock,
    )
    print(ascii_curve(levels_summary))
    print(f"json_path,{json_path}")
    print(f"curve_csv,{curve_path}")


if __name__ == "__main__":
    main()
