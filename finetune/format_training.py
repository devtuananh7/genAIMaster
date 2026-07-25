"""
Step 4 — Format dữ liệu huấn luyện cho LoRA fine-tune.

Chuyển filtered_pairs.json thành JSONL phù hợp cho cả:
  - mlx_lm (Mac Apple Silicon M4)
  - peft/SFTTrainer (PC RTX 4060)

Chia train/valid 90/10 với seed cố định.

Usage:
    python -m finetune.format_training
    python -m finetune.format_training --split-ratio 0.9
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
_MODULE_DIR = Path(__file__).resolve().parent
_DEFAULT_INPUT = _MODULE_DIR / "data" / "filtered_pairs.json"
_DEFAULT_TRAIN = _MODULE_DIR / "data" / "train.jsonl"
_DEFAULT_VALID = _MODULE_DIR / "data" / "valid.jsonl"

SEED = 5410
SPLIT_RATIO = 0.9  # 90% train, 10% valid

# ── Prompt Template ──────────────────────────────────────────────────────────
# Format chuẩn cho instruction fine-tuning base model.
# Dùng cùng template khi đánh giá để đảm bảo nhất quán.

INSTRUCTION_TEMPLATE = """\
### Instruction:
{problem}

Write a Python function named `{entry_point}`.

### Response:
```python
{solution}
```"""


def _format_sample(pair: dict) -> str:
    """Chuyển 1 cặp dữ liệu thành chuỗi huấn luyện."""
    return INSTRUCTION_TEMPLATE.format(
        problem=pair["problem"].strip(),
        entry_point=pair["entry_point"].strip(),
        solution=pair["solution"].strip(),
    )


def format_data(
    input_file: Path = _DEFAULT_INPUT,
    train_file: Path = _DEFAULT_TRAIN,
    valid_file: Path = _DEFAULT_VALID,
    *,
    split_ratio: float = SPLIT_RATIO,
) -> tuple[int, int]:
    """Format và chia dữ liệu train/valid."""
    with input_file.open("r", encoding="utf-8") as f:
        pairs = json.load(f)

    print(f"═══ Formatting {len(pairs)} filtered pairs ═══", file=sys.stderr)

    # Shuffle với seed cố định
    rng = random.Random(SEED)
    indices = list(range(len(pairs)))
    rng.shuffle(indices)

    # Chia train/valid
    split_point = int(len(indices) * split_ratio)
    train_indices = indices[:split_point]
    valid_indices = indices[split_point:]

    # Ghi train.jsonl
    train_file.parent.mkdir(parents=True, exist_ok=True)
    with train_file.open("w", encoding="utf-8") as f:
        for idx in train_indices:
            text = _format_sample(pairs[idx])
            line = json.dumps({"text": text}, ensure_ascii=False)
            f.write(line + "\n")

    # Ghi valid.jsonl
    with valid_file.open("w", encoding="utf-8") as f:
        for idx in valid_indices:
            text = _format_sample(pairs[idx])
            line = json.dumps({"text": text}, ensure_ascii=False)
            f.write(line + "\n")

    print(f"  Train: {len(train_indices)} samples → {train_file}", file=sys.stderr)
    print(f"  Valid: {len(valid_indices)} samples → {valid_file}", file=sys.stderr)

    # Preview 1 mẫu
    if pairs:
        print(f"\n── Preview (sample 0) ──", file=sys.stderr)
        preview = _format_sample(pairs[0])
        for line in preview.splitlines()[:10]:
            print(f"  {line}", file=sys.stderr)
        if len(preview.splitlines()) > 10:
            print(f"  ... ({len(preview.splitlines())} lines total)", file=sys.stderr)

    return len(train_indices), len(valid_indices)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Format dữ liệu cho LoRA fine-tune (OSS-Instruct Step 4).",
    )
    parser.add_argument("--input", default=str(_DEFAULT_INPUT),
                        help="File filtered pairs")
    parser.add_argument("--train", default=str(_DEFAULT_TRAIN),
                        help="Output train JSONL")
    parser.add_argument("--valid", default=str(_DEFAULT_VALID),
                        help="Output valid JSONL")
    parser.add_argument("--split-ratio", type=float, default=SPLIT_RATIO,
                        help="Tỷ lệ train (default: 0.9)")
    args = parser.parse_args()

    format_data(
        input_file=Path(args.input),
        train_file=Path(args.train),
        valid_file=Path(args.valid),
        split_ratio=args.split_ratio,
    )


if __name__ == "__main__":
    main()
