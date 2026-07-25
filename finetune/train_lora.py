"""
finetune/train_lora.py  — Stage B (chạy trên DESKTOP RTX 3080, CUDA)
====================================================================
QLoRA fine-tune deepseek-coder-1.3b-BASE trên train.jsonl.

Cách tiếp cận (POC, hợp RTX 3080 10-12GB):
  - Load base ở 4-bit (bitsandbytes NF4)  -> base đóng băng, ~3-4GB VRAM
  - Gắn LoRA adapter (r=16) -> chỉ train ~0.1-1% tham số
  - SFT causal-LM trên trường "text", 2-3 epoch

Yêu cầu (cài trên desktop, xem requirements-gpu.txt):
  torch (CUDA build), transformers, peft, bitsandbytes, datasets, accelerate

Output: thư mục adapter LoRA (--output), dùng ở eval_mbpp.py qua --adapter.
KHÔNG hợp nhất vào base -> giữ đối chứng before/after sạch (bật/tắt adapter).
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA fine-tune 1.3b-base (CUDA).")
    parser.add_argument("--base-model", default="deepseek-ai/deepseek-coder-1.3b-base")
    parser.add_argument("--train-file", default="data/finetune/train.jsonl")
    parser.add_argument("--output", default="finetune/adapters/deepseek-1.3b-base-lora")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=5410)
    args = parser.parse_args()

    # Import nặng đặt trong main() để file vẫn import/lint được trên máy KHÔNG có CUDA.
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files=args.train_file, split="train")

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_seq_len,
            padding=False,
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output + "-run",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        seed=args.seed,
        optim="paged_adamw_8bit",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
    )
    trainer.train()

    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"saved LoRA adapter -> {args.output}")


if __name__ == "__main__":
    main()
