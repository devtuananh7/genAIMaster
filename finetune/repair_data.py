"""
Step 3b — Sửa lỗi dữ liệu bằng vòng lặp self-repair.

Lấy các mẫu FAIL từ raw_pairs.json, gửi lỗi cụ thể lại cho teacher model
để sửa (tối đa MAX_REPAIR_ATTEMPTS lần). Giống kỹ thuật Reflexion nhưng áp
dụng cho pha sinh dữ liệu huấn luyện.

Quy trình:
  1. Đọc raw_pairs.json
  2. Chạy từng mẫu qua Sandbox Executor
  3. Nếu PASS → giữ nguyên
  4. Nếu FAIL → gửi lỗi cho teacher → nhận bản sửa → chạy lại
  5. Lặp tối đa 2 lần sửa
  6. Ghi kết quả ra repaired_pairs.json (thay thế filtered_pairs.json)

Usage:
    python -m finetune.repair_data
    python -m finetune.repair_data --max-attempts 3
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests

from harness.executor import run as sandbox_run
from harness.types import ExecutionResult, Task

# ── Paths ────────────────────────────────────────────────────────────────────
_MODULE_DIR = Path(__file__).resolve().parent
_DEFAULT_INPUT = _MODULE_DIR / "data" / "raw_pairs.json"
_DEFAULT_OUTPUT = _MODULE_DIR / "data" / "filtered_pairs.json"
_DEFAULT_STATS = _MODULE_DIR / "data" / "filter_stats.json"

# ── Teacher Config (tái sử dụng từ generate_data) ───────────────────────────
TEACHER_MODEL = "qwen3.5:9b"
TEACHER_TEMPERATURE = 0.4  # thấp hơn khi sửa lỗi → ít sáng tạo, chính xác hơn
TEACHER_MAX_TOKENS = 2048
REQUEST_TIMEOUT = 180
LOCAL_OLLAMA_URL = "http://localhost:11434"
MAX_REPAIR_ATTEMPTS = 2


def _resolve_url(base_url: str | None = None) -> str:
    import os
    return (base_url or os.environ.get("OLLAMA_HOST") or LOCAL_OLLAMA_URL).rstrip("/")


# ── Repair Prompts ───────────────────────────────────────────────────────────

REPAIR_SYSTEM = """\
You are a Python debugging expert. You will receive a Python function that has \
errors, along with the exact error message. Fix the function so it runs correctly \
and passes all test assertions.

Rules:
- Fix ONLY the solution function (do NOT change the tests unless they are clearly wrong)
- Use ONLY the Python standard library
- The function must be fully self-contained
- If tests are wrong, fix them too but keep the problem intent\
"""

REPAIR_USER_TEMPLATE = """\
Problem: {problem}
Function name: {entry_point}

Current solution (BROKEN):
```python
{solution}
```

Tests:
{tests_str}

Error type: {error_type}
Error details:
```
{error_details}
```

Fix the solution (and tests if needed). /no_think
Respond with ONLY valid JSON:
{{"solution": "def {entry_point}(...):\\n    ...", "tests": ["assert ...", "assert ..."]}}"""


# ── Helpers ──────────────────────────────────────────────────────────────────

def _call_teacher(
    system: str,
    user: str,
    *,
    base_url: str,
    model: str = TEACHER_MODEL,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {
            "temperature": TEACHER_TEMPERATURE,
            "num_predict": TEACHER_MAX_TOKENS,
        },
        "format": "json",
    }
    for attempt in range(3):
        try:
            resp = requests.post(
                f"{base_url}/api/chat", json=payload, timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            return str(resp.json().get("message", {}).get("content", ""))
        except (requests.Timeout, requests.ConnectionError) as exc:
            time.sleep(2 ** (attempt + 1))
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status and 400 <= status < 500:
                raise
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError("Teacher unreachable after 3 retries")


def _parse_repair(raw: str) -> dict[str, Any] | None:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or "solution" not in data:
        return None
    return data


def _pair_to_task(pair: dict) -> Task:
    return Task(
        task_id=-1,
        text=pair["problem"],
        test_list=pair["tests"],
        test_imports=[],
    )


def _run_and_check(pair: dict) -> tuple[ExecutionResult, str]:
    """Chạy pair qua executor, trả kết quả + error detail."""
    task = _pair_to_task(pair)
    try:
        result = sandbox_run(pair["solution"], task)
    except Exception as exc:
        result = ExecutionResult(
            status="error_runtime", stdout="", stderr=str(exc),
            traceback=str(exc), failed_test=None,
            passed_count=0, total_count=len(pair["tests"]), duration_ms=0,
        )

    error_detail = ""
    if result.status != "pass":
        parts = []
        if result.traceback:
            parts.append(result.traceback)
        if result.failed_test:
            parts.append(f"Failed test: {result.failed_test}")
        if result.stderr and result.stderr not in (result.traceback or ""):
            parts.append(result.stderr[:300])
        error_detail = "\n".join(parts) or f"Status: {result.status}"

    return result, error_detail


# ── Main Pipeline ────────────────────────────────────────────────────────────

def repair(
    input_file: Path = _DEFAULT_INPUT,
    output_file: Path = _DEFAULT_OUTPUT,
    stats_file: Path = _DEFAULT_STATS,
    *,
    base_url: str | None = None,
    model: str = TEACHER_MODEL,
    max_attempts: int = MAX_REPAIR_ATTEMPTS,
) -> list[dict]:
    resolved_url = _resolve_url(base_url)

    with input_file.open("r", encoding="utf-8") as f:
        raw_pairs = json.load(f)

    print(f"═══ Repair Pipeline: {len(raw_pairs)} pairs ═══", file=sys.stderr)
    print(f"  Teacher: {model} @ {resolved_url}", file=sys.stderr)
    print(f"  Max repair attempts: {max_attempts}", file=sys.stderr)

    filtered: list[dict] = []
    stats = {
        "total": len(raw_pairs),
        "pass_direct": 0,
        "pass_after_repair": 0,
        "fail_final": 0,
        "repair_attempts": 0,
    }

    for idx, pair in enumerate(raw_pairs):
        progress = f"[{idx+1}/{len(raw_pairs)}]"
        entry = pair.get("entry_point", "?")

        # ── Lần chạy đầu ────────────────────────────────────────────────
        result, error_detail = _run_and_check(pair)

        if result.status == "pass":
            stats["pass_direct"] += 1
            pair["repair_rounds"] = 0
            filtered.append(pair)
            print(f"  {progress} {entry} — ✓ pass (direct)", file=sys.stderr)
            continue

        # ── Vòng lặp sửa lỗi ────────────────────────────────────────────
        current_pair = dict(pair)
        repaired = False

        for attempt in range(1, max_attempts + 1):
            stats["repair_attempts"] += 1
            print(
                f"  {progress} {entry} — repair {attempt}/{max_attempts} "
                f"({result.status}) ... ",
                end="", file=sys.stderr,
            )

            try:
                tests_str = "\n".join(current_pair["tests"])
                prompt = REPAIR_USER_TEMPLATE.format(
                    problem=current_pair["problem"],
                    entry_point=current_pair["entry_point"],
                    solution=current_pair["solution"],
                    tests_str=tests_str,
                    error_type=result.status,
                    error_details=error_detail,
                )
                raw_resp = _call_teacher(
                    REPAIR_SYSTEM, prompt, base_url=resolved_url, model=model,
                )
                parsed = _parse_repair(raw_resp)

                if parsed is None:
                    print("✗ parse fail", file=sys.stderr)
                    continue

                # Cập nhật solution (và tests nếu teacher sửa)
                current_pair["solution"] = parsed["solution"]
                if "tests" in parsed and isinstance(parsed["tests"], list) and parsed["tests"]:
                    current_pair["tests"] = parsed["tests"]

                # Chạy lại
                result, error_detail = _run_and_check(current_pair)

                if result.status == "pass":
                    stats["pass_after_repair"] += 1
                    current_pair["repair_rounds"] = attempt
                    filtered.append(current_pair)
                    repaired = True
                    print("✓ fixed!", file=sys.stderr)
                    break
                else:
                    print(f"✗ still {result.status}", file=sys.stderr)

            except Exception as exc:
                print(f"✗ error: {exc}", file=sys.stderr)

        if not repaired:
            stats["fail_final"] += 1
            print(f"  {progress} {entry} — ✗ gave up after {max_attempts} repairs",
                  file=sys.stderr)

    # ── Save ─────────────────────────────────────────────────────────────
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)
        f.write("\n")

    with stats_file.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
        f.write("\n")

    total_pass = stats["pass_direct"] + stats["pass_after_repair"]
    pass_rate = total_pass / stats["total"] * 100 if stats["total"] > 0 else 0

    print(f"\n{'═'*50}", file=sys.stderr)
    print(f"  KẾT QUẢ REPAIR PIPELINE", file=sys.stderr)
    print(f"{'═'*50}", file=sys.stderr)
    print(f"  Total:            {stats['total']}", file=sys.stderr)
    print(f"  ✓ Pass (direct):  {stats['pass_direct']}", file=sys.stderr)
    print(f"  ✓ Pass (repaired):{stats['pass_after_repair']}", file=sys.stderr)
    print(f"  ✓ Total pass:     {total_pass} ({pass_rate:.1f}%)", file=sys.stderr)
    print(f"  ✗ Final fail:     {stats['fail_final']}", file=sys.stderr)
    print(f"  Repair attempts:  {stats['repair_attempts']}", file=sys.stderr)
    print(f"  Output:           {output_file}", file=sys.stderr)

    return filtered


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sửa lỗi dữ liệu bằng vòng lặp self-repair (OSS-Instruct Step 3b).",
    )
    parser.add_argument("--input", default=str(_DEFAULT_INPUT))
    parser.add_argument("--output", default=str(_DEFAULT_OUTPUT))
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=TEACHER_MODEL)
    parser.add_argument("--max-attempts", type=int, default=MAX_REPAIR_ATTEMPTS,
                        help="Số lần sửa tối đa mỗi mẫu (default: 2)")
    args = parser.parse_args()

    repair(
        input_file=Path(args.input),
        output_file=Path(args.output),
        base_url=args.base_url,
        model=args.model,
        max_attempts=args.max_attempts,
    )


if __name__ == "__main__":
    main()
