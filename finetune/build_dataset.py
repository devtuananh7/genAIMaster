"""
finetune/build_dataset.py  — Stage A, bước 3 (đóng gói)
=======================================================
Chuyển các mẫu đã lọc thành train.jsonl để SFT/QLoRA.

Định dạng prompt PHẢI KHỚP với lúc eval (finetune/eval_mbpp.py) để đối chứng công
bằng: cùng khung "### System / ### User / ### Assistant" như harness.hf_client, lời
giải bọc trong ```python ... ```. Nhờ vậy 1.3b-base học đúng phong cách sinh code mà
eval mong đợi.

Mỗi dòng train.jsonl: {"text": "<prompt + completion + EOS>"}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SYSTEM = (
    "You are an expert Python programmer. You write correct, runnable code. "
    "Return ONLY the function inside a single ```python code block."
)


def format_example(sample: dict[str, Any], eos: str = "") -> str:
    problem = sample["problem"].strip()
    tests = "\n".join(sample.get("tests", []))
    solution = sample["solution"].strip()
    user = (
        f"{problem}\n\n"
        f"It must pass these tests:\n{tests}\n\n"
        f"Return only the function inside a single ```python code block."
    )
    completion = f"```python\n{solution}\n```"
    return (
        f"### System:\n{SYSTEM}\n\n"
        f"### User:\n{user}\n\n"
        f"### Assistant:\n{completion}{eos}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SFT train.jsonl from filtered samples.")
    parser.add_argument("--input", default="data/finetune/teacher_filtered.jsonl")
    parser.add_argument("--output", default="data/finetune/train.jsonl")
    parser.add_argument("--eos", default="", help="EOS token nối cuối completion (tuỳ tokenizer)")
    args = parser.parse_args()

    samples = [json.loads(l) for l in Path(args.input).read_text().splitlines() if l.strip()]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(json.dumps({"text": format_example(sample, args.eos)}, ensure_ascii=False) + "\n")
    print(f"wrote {len(samples)} training rows -> {out}")


if __name__ == "__main__":
    main()
