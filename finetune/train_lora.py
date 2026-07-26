"""
Step 5 — LoRA Fine-tune deepseek-coder-1.3b-base.

Hỗ trợ 2 backend:
  - mlx:  Mac Apple Silicon M4 (16GB RAM, 7 GPU cores) — dùng mlx_lm
  - peft: PC RTX 4060 8GB VRAM — dùng peft + transformers (QLoRA 4-bit)

Usage:
    # Mac M4:
    python -m finetune.train_lora --backend mlx

    # PC RTX 4060:
    python -m finetune.train_lora --backend peft

    # Custom config:
    python -m finetune.train_lora --backend mlx --epochs 3 --lr 2e-4 --batch-size 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
_MODULE_DIR = Path(__file__).resolve().parent
_DATA_DIR = _MODULE_DIR / "data"
_ADAPTERS_DIR = _MODULE_DIR / "adapters"

# ── Default Config ───────────────────────────────────────────────────────────
BASE_MODEL = "deepseek-ai/deepseek-coder-1.3b-base"  # HuggingFace ID
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj"]  # attention projections

DEFAULT_EPOCHS = 3
DEFAULT_LR = 2e-4
DEFAULT_BATCH_SIZE = 4
DEFAULT_MAX_SEQ_LEN = 512
DEFAULT_GRAD_ACCUM = 2  # effective batch = batch_size * grad_accum = 8


# ══════════════════════════════════════════════════════════════════════════════
# BACKEND: MLX (Mac Apple Silicon)
# ══════════════════════════════════════════════════════════════════════════════

def train_mlx(
    *,
    model: str = BASE_MODEL,
    data_dir: Path = _DATA_DIR,
    output_dir: Path = _ADAPTERS_DIR / "mlx",
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    lora_rank: int = LORA_RANK,
    lora_layers: int = 16,
) -> None:
    """Train LoRA trên Mac Apple Silicon bằng mlx_lm."""
    try:
        from mlx_lm import lora as mlx_lora  # noqa: F811
    except ImportError:
        print(
            "ERROR: mlx_lm chưa cài. Chạy:\n"
            "  pip install -r finetune/requirements_mac.txt",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Tính số iterations từ epochs
    train_file = data_dir / "train.jsonl"
    if train_file.exists():
        with train_file.open("r") as f:
            n_samples = sum(1 for _ in f)
    else:
        print(f"ERROR: {train_file} not found. Run format_training.py first.", file=sys.stderr)
        sys.exit(1)

    steps_per_epoch = max(1, n_samples // batch_size)
    total_iters = steps_per_epoch * epochs

    print(f"═══ MLX LoRA Training ═══", file=sys.stderr)
    print(f"  Model:       {model}", file=sys.stderr)
    print(f"  Data:        {data_dir}", file=sys.stderr)
    print(f"  Samples:     {n_samples} train", file=sys.stderr)
    print(f"  Epochs:      {epochs} ({total_iters} iterations)", file=sys.stderr)
    print(f"  Batch size:  {batch_size}", file=sys.stderr)
    print(f"  LoRA rank:   {lora_rank}", file=sys.stderr)
    print(f"  LR:          {lr}", file=sys.stderr)
    print(f"  Output:      {output_dir}", file=sys.stderr)

    # Lưu config
    config = {
        "model": model,
        "backend": "mlx",
        "epochs": epochs,
        "iterations": total_iters,
        "batch_size": batch_size,
        "lr": lr,
        "lora_rank": lora_rank,
        "lora_layers": lora_layers,
        "max_seq_len": max_seq_len,
        "n_train_samples": n_samples,
    }
    with (output_dir / "train_config.json").open("w") as f:
        json.dump(config, f, indent=2)

    # Chạy mlx_lm.lora training
    # mlx_lm cung cấp hàm lora.train() hoặc CLI
    # Dùng CLI thông qua subprocess cho đơn giản và tương thích
    import subprocess
    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", model,
        "--data", str(data_dir),
        "--train",
        "--batch-size", str(batch_size),
        "--lora-layers", str(lora_layers),
        "--iters", str(total_iters),
        "--learning-rate", str(lr),
        "--adapter-path", str(output_dir),
        "--max-seq-length", str(max_seq_len),
        "--val-batches", "5",
        "--steps-per-eval", str(steps_per_epoch),
        "--steps-per-report", "10",
    ]

    print(f"\n  Command: {' '.join(cmd)}\n", file=sys.stderr)
    result = subprocess.run(cmd, check=False)

    if result.returncode == 0:
        print(f"\n✓ Training complete! Adapters saved to {output_dir}", file=sys.stderr)
    else:
        print(f"\n✗ Training failed with exit code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)


# ══════════════════════════════════════════════════════════════════════════════
# BACKEND: PEFT (PC RTX 4060 — QLoRA 4-bit)
# ══════════════════════════════════════════════════════════════════════════════

def train_peft(
    *,
    model: str = BASE_MODEL,
    data_dir: Path = _DATA_DIR,
    output_dir: Path = _ADAPTERS_DIR / "peft",
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    grad_accum: int = DEFAULT_GRAD_ACCUM,
    lora_rank: int = LORA_RANK,
    lora_alpha: int = LORA_ALPHA,
    lora_dropout: float = LORA_DROPOUT,
) -> None:
    """Train QLoRA trên PC RTX 4060 bằng peft + transformers."""
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
        )
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        print(
            f"ERROR: Missing dependency: {exc}\n"
            "  Chạy: pip install -r finetune/requirements_pc.txt",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"═══ PEFT QLoRA Training ═══", file=sys.stderr)
    print(f"  Model:       {model}", file=sys.stderr)
    print(f"  Device:      {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", file=sys.stderr)
    print(f"  VRAM:        {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if torch.cuda.is_available() else "  VRAM:        N/A", file=sys.stderr)
    print(f"  Epochs:      {epochs}", file=sys.stderr)
    print(f"  Batch:       {batch_size} × {grad_accum} = {batch_size * grad_accum} effective", file=sys.stderr)
    print(f"  LoRA:        r={lora_rank}, alpha={lora_alpha}", file=sys.stderr)
    print(f"  LR:          {lr}", file=sys.stderr)
    print(f"  Output:      {output_dir}", file=sys.stderr)

    # ── QLoRA: 4-bit quantization ────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # ── Load model + tokenizer ───────────────────────────────────────────
    print("\n  Loading model...", file=sys.stderr)
    base_model = AutoModelForCausalLM.from_pretrained(
        model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model = prepare_model_for_kbit_training(base_model)

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── LoRA config ──────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=TARGET_MODULES,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    peft_model = get_peft_model(base_model, lora_config)
    trainable, total = peft_model.get_nb_trainable_parameters()
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)",
          file=sys.stderr)

    # ── Load dataset ─────────────────────────────────────────────────────
    train_path = str(data_dir / "train.jsonl")
    valid_path = str(data_dir / "valid.jsonl")
    dataset = load_dataset("json", data_files={"train": train_path, "validation": valid_path})

    # ── Training arguments ───────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        max_length=max_seq_len,
        dataset_text_field="text",
        packing=False,
        report_to="none",  # không dùng wandb
    )

    # ── Train ────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )

    print("\n  Starting training...", file=sys.stderr)
    trainer.train()

    # ── Save ─────────────────────────────────────────────────────────────
    peft_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Lưu config
    config = {
        "model": model,
        "backend": "peft",
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "lr": lr,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "target_modules": TARGET_MODULES,
        "max_seq_len": max_seq_len,
        "quantization": "4bit-nf4",
    }
    with (output_dir / "train_config.json").open("w") as f:
        json.dump(config, f, indent=2)

    print(f"\n✓ Training complete! Adapters saved to {output_dir}", file=sys.stderr)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune deepseek-coder-1.3b-base (OSS-Instruct Step 5).",
    )
    parser.add_argument("--backend", required=True, choices=["mlx", "peft"],
                        help="mlx = Mac M4 (mlx_lm) | peft = PC RTX 4060 (QLoRA)")
    parser.add_argument("--model", default=BASE_MODEL,
                        help=f"HuggingFace model ID (default: {BASE_MODEL})")
    parser.add_argument("--data-dir", default=str(_DATA_DIR),
                        help="Thư mục chứa train.jsonl / valid.jsonl")
    parser.add_argument("--output-dir", default=None,
                        help="Thư mục lưu adapter (default: adapters/<backend>)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--lora-rank", type=int, default=LORA_RANK)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else _ADAPTERS_DIR / args.backend

    if args.backend == "mlx":
        train_mlx(
            model=args.model,
            data_dir=data_dir,
            output_dir=output_dir,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            lora_rank=args.lora_rank,
        )
    else:
        train_peft(
            model=args.model,
            data_dir=data_dir,
            output_dir=output_dir,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            lora_rank=args.lora_rank,
        )


if __name__ == "__main__":
    main()
