"""
finetune/eval_mbpp.py  — Stage C (chạy trên DESKTOP RTX 3080)
=============================================================
Đo pass@1 của 1.3b-BASE TRƯỚC vs SAU fine-tune, trên ĐÚNG 50 bài MBPP của M0.

  before : chạy base thuần            (không --adapter)
  after  : chạy base + LoRA adapter   (--adapter <đường dẫn>)

So sánh HỢP LỆ DUY NHẤT: base-before vs base-after. KHÔNG so với instruct (M1-M4).
Tái dùng harness M0: loader (50 bài), extractor, executor, và prompt style baseline.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from harness.extractor import extract_code
from harness.executor import run as run_executor
from harness.loader import load_tasks
from harness.signature import entry_function_name
from harness.types import Task

SYSTEM = (
    "You are an expert Python programmer. You write correct, runnable code. "
    "Return ONLY the function inside a single ```python code block."
)


def build_prompt(task: Task) -> str:
    tests = "\n".join(task.test_list)
    func = entry_function_name(task.test_list)
    name_line = f"Write a Python function named `{func}`. " if func else "Write a Python function. "
    tail = (
        f"Return only the function inside a single ```python code block, using exactly the name `{func}`."
        if func
        else "Return only the function inside a single ```python code block."
    )
    user = f"{task.text}\n\n{name_line}It must pass these tests:\n{tests}\n\n{tail}"
    return f"### System:\n{SYSTEM}\n\n### User:\n{user}\n\n### Assistant:\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval 1.3b-base pass@1 before/after (CUDA).")
    parser.add_argument("--base-model", default="deepseek-ai/deepseek-coder-1.3b-base")
    parser.add_argument("--adapter", default=None, help="đường dẫn LoRA adapter (bỏ trống = before)")
    parser.add_argument("--tasks", default="data/selected_tasks.json")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--label", default=None, help="nhãn ghi kết quả (before/after)")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    label = args.label or ("after" if args.adapter else "before")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb, device_map="auto", trust_remote_code=True
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    tasks = load_tasks(args.tasks)
    per_task = []
    passed = 0
    print("task_id,status")
    for task in tasks:
        prompt = build_prompt(task)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=max(args.temperature, 0.01),
                pad_token_id=tokenizer.pad_token_id,
            )
        gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        code = extract_code(gen)
        result = run_executor(code, task)
        status = result.status
        passed += int(status == "pass")
        per_task.append({"task_id": task.task_id, "status": status, "code": code})
        print(f"{task.task_id},{status}")

    pass1 = passed / len(tasks) if tasks else 0.0
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("results") / "finetune"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{label}_{ts}.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "label": label,
                "base_model": args.base_model,
                "adapter": args.adapter,
                "timestamp": ts,
                "total_pass1": pass1,
                "total_tasks": len(tasks),
                "per_task": per_task,
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n{label}: pass@1 = {pass1:.4f} ({passed}/{len(tasks)})")
    print(f"json_path,{out_path}")


if __name__ == "__main__":
    main()
