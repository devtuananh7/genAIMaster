"""
finetune/gen_teacher.py  — Stage A, bước 2
==========================================
Với mỗi seed, gọi TEACHER (OSS-Instruct thu nhỏ) sinh một bộ ba:
  { "problem": <đề bài NL>, "solution": <hàm Python>, "tests": [<assert>...] }

`tests` để bước lọc (filter_runnable) kiểm chạy được bằng Executor M0.
Teacher được yêu cầu trả JSON để parse ổn định.

Nếu teacher chỉ là 1.3b/6.7b (yếu) -> thực chất là "self-instruct", GHI RÕ hạn chế
(spec M5). Nên dùng model mạnh: deepseek-chat, gpt-4o-mini, qwen2.5-coder-32b...
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Callable

from finetune.teacher_client import chat as teacher_chat
from finetune.teacher_client import resolve_model

TEACHER_SYSTEM = (
    "You are a Python programming instructor creating training data. "
    "Given a snippet of real code as inspiration, invent ONE small, self-contained "
    "programming task and its correct solution. Respond ONLY with a JSON object."
)

TEACHER_USER_TEMPLATE = """Here is a snippet of real Python code for inspiration:

```python
{snippet}
```

Create ONE small self-contained programming exercise inspired by it. Respond with a
single JSON object with EXACTLY these keys:
- "problem": a clear one-paragraph task description (natural language).
- "solution": a single self-contained Python function that solves it (string, no markdown fences).
- "tests": a list of 2-3 Python `assert` statements that call the solution function.

The solution must be runnable standalone (only stdlib imports). JSON only, no prose."""

JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_teacher_json(raw: str) -> dict[str, Any] | None:
    match = JSON_BLOCK_RE.search(raw)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not all(k in obj for k in ("problem", "solution", "tests")):
        return None
    if not isinstance(obj["tests"], list) or not obj["tests"]:
        return None
    return obj


def generate_for_seed(
    seed: dict[str, Any],
    *,
    temperature: float,
    max_tokens: int,
    chat_fn: Callable[..., str],
    model: str | None,
) -> dict[str, Any] | None:
    user = TEACHER_USER_TEMPLATE.format(snippet=seed["snippet"])
    raw = chat_fn(TEACHER_SYSTEM, user, temperature=temperature, max_tokens=max_tokens, model=model)
    parsed = _parse_teacher_json(raw)
    if parsed is None:
        return None
    return {
        "seed_id": seed["seed_id"],
        "source_repo": seed["source_repo"],
        "problem": str(parsed["problem"]).strip(),
        "solution": str(parsed["solution"]).strip(),
        "tests": [str(t).strip() for t in parsed["tests"]],
    }


def make_mock_chat() -> Callable[..., str]:
    """Mock teacher — CHỈ để self-test plumbing khi chưa có API key."""

    def mock_chat(system: str, user: str, *, temperature=0.7, max_tokens=1024, model=None, **kw):
        obj = {
            "problem": "Write a function `double_all(nums)` that returns each number doubled.",
            "solution": "def double_all(nums):\n    return [n * 2 for n in nums]",
            "tests": ["assert double_all([1, 2, 3]) == [2, 4, 6]", "assert double_all([]) == []"],
        }
        return json.dumps(obj)

    return mock_chat


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate teacher (problem, solution, tests).")
    parser.add_argument("--seeds", default="data/finetune/seeds.jsonl")
    parser.add_argument("--output", default="data/finetune/teacher_raw.jsonl")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--model")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    seeds = [json.loads(line) for line in Path(args.seeds).read_text().splitlines() if line.strip()]
    if args.limit:
        seeds = seeds[: args.limit]

    chat_fn = make_mock_chat() if args.mock else teacher_chat
    model = "MOCK" if args.mock else resolve_model(args.model)
    print(f"teacher={model}  seeds={len(seeds)}  mock={args.mock}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with out.open("w", encoding="utf-8") as fh:
        for i, seed in enumerate(seeds, 1):
            try:
                sample = generate_for_seed(
                    seed,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    chat_fn=chat_fn,
                    model=args.model,
                )
            except Exception as exc:  # noqa: BLE001 - log & tiếp tục
                print(f"  [{i}/{len(seeds)}] {seed['seed_id']} ERROR: {type(exc).__name__}")
                continue
            if sample is None:
                continue
            fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
            kept += 1
    print(f"generated {kept}/{len(seeds)} valid samples -> {out}")


if __name__ == "__main__":
    main()
