"""
Step 2 — Sinh dữ liệu huấn luyện bằng model giáo viên (qwen3.5:9b).

Với mỗi seed snippet, yêu cầu giáo viên tạo cặp (đề bài, lời giải) —
ý tưởng cốt lõi của OSS-Instruct.

Teacher model: qwen3.5:9b qua Ollama tại server local.

Usage:
    python -m finetune.generate_data
    python -m finetune.generate_data --limit 10          # chạy thử 10 mẫu
    python -m finetune.generate_data --resume             # tiếp tục từ checkpoint
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

# ── Paths & Config ───────────────────────────────────────────────────────────
_MODULE_DIR = Path(__file__).resolve().parent
_DEFAULT_SEEDS = _MODULE_DIR / "data" / "seed_snippets.json"
_DEFAULT_OUTPUT = _MODULE_DIR / "data" / "raw_pairs.json"
_CHECKPOINT_FILE = _MODULE_DIR / "data" / ".generate_checkpoint.json"

# Teacher model qua Ollama LOCAL (chạy trên Mac, KHÔNG phải server dự án)
TEACHER_MODEL = "qwen3.5:9b"
TEACHER_TEMPERATURE = 0.7
TEACHER_MAX_TOKENS = 2048
REQUEST_TIMEOUT = 180  # giáo viên 9b cần thời gian
LOCAL_OLLAMA_URL = "http://localhost:11434"


def _resolve_teacher_url(base_url: str | None = None) -> str:
    """Resolve URL cho teacher model — mặc định localhost (KHÔNG dùng server dự án)."""
    import os
    return (base_url or os.environ.get("OLLAMA_HOST") or LOCAL_OLLAMA_URL).rstrip("/")

# ── Teacher Prompt ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a programming instructor creating Python exercises.
You will receive a code snippet from an open-source project.
Your task: create an ORIGINAL, self-contained programming exercise inspired by
the concepts, patterns, or techniques in the snippet.

Rules:
- The problem must be solvable with a SINGLE Python function
- Use ONLY the Python standard library (no numpy, pandas, etc.)
- The function must be fully self-contained
- Provide 2-3 test assertions that verify correctness
- Tests must use simple, deterministic values
- Do NOT copy the snippet directly — create an original problem\
"""

USER_PROMPT_TEMPLATE = """\
Code snippet from repository "{repo}":

```python
{code}
```

Create a programming exercise inspired by this code. /no_think

Respond with ONLY valid JSON (no markdown, no explanation):
{{"problem": "Write a function to ...", "entry_point": "function_name", "solution": "def function_name(...):\\n    ...", "tests": ["assert function_name(...) == ...", "assert function_name(...) == ..."]}}"""


# ── Helpers ──────────────────────────────────────────────────────────────────

def _call_teacher(
    system_prompt: str,
    user_prompt: str,
    *,
    base_url: str,
    model: str = TEACHER_MODEL,
) -> str:
    """Gọi teacher model qua Ollama API."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "temperature": TEACHER_TEMPERATURE,
            "num_predict": TEACHER_MAX_TOKENS,
        },
        "format": "json",  # Ollama JSON mode
    }

    for attempt in range(3):
        try:
            resp = requests.post(
                f"{base_url}/api/chat",
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            return str(data.get("message", {}).get("content", ""))
        except (requests.Timeout, requests.ConnectionError) as exc:
            wait = 2 ** (attempt + 1)
            print(f"    [retry {attempt+1}] {exc} — wait {wait}s", file=sys.stderr)
            time.sleep(wait)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status and 400 <= status < 500:
                raise
            wait = 2 ** (attempt + 1)
            print(f"    [retry {attempt+1}] HTTP {status} — wait {wait}s", file=sys.stderr)
            time.sleep(wait)

    raise RuntimeError("Teacher model unreachable after 3 retries")


def _parse_response(raw: str) -> dict[str, Any] | None:
    """Parse JSON response từ teacher, trả None nếu không hợp lệ."""
    # Loại bỏ markdown fences nếu có
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return None

    # Validate required fields
    required = {"problem", "entry_point", "solution", "tests"}
    if not isinstance(data, dict) or not required.issubset(data.keys()):
        return None
    if not isinstance(data["tests"], list) or len(data["tests"]) < 1:
        return None
    if not data["solution"].strip():
        return None

    return data


def _load_checkpoint(checkpoint_file: Path) -> set[int]:
    """Load danh sách index đã xử lý."""
    if not checkpoint_file.exists():
        return set()
    with checkpoint_file.open("r", encoding="utf-8") as f:
        return set(json.load(f))


def _save_checkpoint(checkpoint_file: Path, processed: set[int]) -> None:
    """Lưu checkpoint."""
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_file.open("w", encoding="utf-8") as f:
        json.dump(sorted(processed), f)


# ── Main Pipeline ────────────────────────────────────────────────────────────

def generate(
    seeds_file: Path = _DEFAULT_SEEDS,
    output: Path = _DEFAULT_OUTPUT,
    *,
    base_url: str | None = None,
    model: str = TEACHER_MODEL,
    limit: int | None = None,
    resume: bool = False,
) -> list[dict]:
    """Sinh dữ liệu huấn luyện từ seed snippets."""
    resolved_url = _resolve_teacher_url(base_url)

    # Load seeds
    with seeds_file.open("r", encoding="utf-8") as f:
        seeds = json.load(f)

    if limit is not None:
        seeds = seeds[:limit]

    # Resume support
    processed_indices: set[int] = set()
    existing_pairs: list[dict] = []

    if resume:
        processed_indices = _load_checkpoint(_CHECKPOINT_FILE)
        if output.exists():
            with output.open("r", encoding="utf-8") as f:
                existing_pairs = json.load(f)
        print(f"  Resuming: {len(processed_indices)} already processed", file=sys.stderr)

    pairs = list(existing_pairs)
    stats = {"total": len(seeds), "success": len(existing_pairs), "parse_fail": 0, "error": 0}

    print(f"═══ Generating data from {len(seeds)} seeds ═══", file=sys.stderr)
    print(f"  Teacher: {model} @ {resolved_url}", file=sys.stderr)
    print(f"  Output: {output}", file=sys.stderr)

    for idx, seed in enumerate(seeds):
        if idx in processed_indices:
            continue

        progress = f"[{idx+1}/{len(seeds)}]"
        func_name = seed.get("function_name", "?")
        print(f"  {progress} {seed['repo']}::{func_name} ... ", end="", file=sys.stderr)

        try:
            prompt = USER_PROMPT_TEMPLATE.format(
                repo=seed["repo"],
                code=seed["code"],
            )
            raw_response = _call_teacher(
                SYSTEM_PROMPT, prompt, base_url=resolved_url, model=model,
            )
            parsed = _parse_response(raw_response)

            if parsed is None:
                print("✗ parse fail", file=sys.stderr)
                stats["parse_fail"] += 1
            else:
                pair = {
                    "seed_index": idx,
                    "seed_repo": seed["repo"],
                    "seed_function": func_name,
                    "seed_code": seed["code"],
                    "problem": parsed["problem"],
                    "entry_point": parsed["entry_point"],
                    "solution": parsed["solution"],
                    "tests": parsed["tests"],
                }
                pairs.append(pair)
                stats["success"] += 1
                print("✓", file=sys.stderr)

        except Exception as exc:
            print(f"✗ error: {exc}", file=sys.stderr)
            stats["error"] += 1

        # Checkpoint mỗi 10 mẫu
        processed_indices.add(idx)
        if (idx + 1) % 10 == 0:
            _save_checkpoint(_CHECKPOINT_FILE, processed_indices)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("w", encoding="utf-8") as f:
                json.dump(pairs, f, ensure_ascii=False, indent=2)
                f.write("\n")

    # Ghi kết quả cuối
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
        f.write("\n")
    _save_checkpoint(_CHECKPOINT_FILE, processed_indices)

    print(f"\n═══ Done ═══", file=sys.stderr)
    print(f"  Success: {stats['success']}/{stats['total']}", file=sys.stderr)
    print(f"  Parse fail: {stats['parse_fail']}", file=sys.stderr)
    print(f"  Error: {stats['error']}", file=sys.stderr)
    print(f"  Output: {output}", file=sys.stderr)

    return pairs


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sinh dữ liệu huấn luyện bằng teacher model (OSS-Instruct Step 2).",
    )
    parser.add_argument("--seeds", default=str(_DEFAULT_SEEDS),
                        help="File seed snippets (default: data/seed_snippets.json)")
    parser.add_argument("--output", default=str(_DEFAULT_OUTPUT),
                        help="File output (default: data/raw_pairs.json)")
    parser.add_argument("--base-url", default=None,
                        help="Ollama base URL (default: from OLLAMA_HOST or contract)")
    parser.add_argument("--model", default=TEACHER_MODEL,
                        help=f"Teacher model (default: {TEACHER_MODEL})")
    parser.add_argument("--limit", type=int, default=None,
                        help="Chỉ xử lý N seed đầu tiên (debug)")
    parser.add_argument("--resume", action="store_true",
                        help="Tiếp tục từ checkpoint (nếu bị gián đoạn)")
    args = parser.parse_args()

    generate(
        seeds_file=Path(args.seeds),
        output=Path(args.output),
        base_url=args.base_url,
        model=args.model,
        limit=args.limit,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
