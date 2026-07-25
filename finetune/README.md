# finetune — M5: QLoRA fine-tune 1.3b-BASE (OSS-Instruct thu nhỏ)

Trục **training-time** cho báo cáo. So sánh **base-TRƯỚC vs base-SAU** fine-tune trên
50 bài MBPP của M0. ⚠️ **KHÔNG** so với instruct (M1-M4) — khác model.

## Pipeline chia theo máy

```
  Stage A (data)   — chạy Ở ĐÂU CŨNG ĐƯỢC (Mac hoặc desktop)
    collect_seeds → gen_teacher (API bên thứ 3) → filter_runnable (Executor M0) → build_dataset
  Stage B (train)  — chạy trên DESKTOP RTX 3080 (CUDA)
    train_lora  (QLoRA 4-bit, LoRA r=16, 2-3 epoch)
  Stage C (eval)   — chạy trên DESKTOP RTX 3080 (CUDA)
    eval_mbpp before (base thuần) vs after (base + adapter)
```

## Stage A — dữ liệu (không cần GPU)

```bash
# 1. thu seed từ repo license mở (mặc định boltons/BSD; thêm --packages more_itertools,toolz)
./.venv/bin/python -m finetune.collect_seeds --packages boltons

# 2. teacher sinh (problem, solution, tests) — cần API bên thứ 3
export TEACHER_BASE_URL=https://api.deepseek.com/v1     # hoặc OpenAI/Together/OpenRouter...
export TEACHER_API_KEY=sk-...
export TEACHER_MODEL=deepseek-chat                       # nên dùng model MẠNH làm teacher
./.venv/bin/python -m finetune.gen_teacher

# 3. lọc mẫu chạy được bằng Executor M0
./.venv/bin/python -m finetune.filter_runnable

# 4. đóng gói train.jsonl (khớp prompt style với eval)
./.venv/bin/python -m finetune.build_dataset

# (self-test không cần API key — dùng teacher mock)
./.venv/bin/python -m finetune.gen_teacher --mock --limit 20
```

## Stage B + C — train & eval (DESKTOP RTX 3080)

```bash
pip install -r finetune/requirements-gpu.txt            # torch bản CUDA (xem file)

# eval TRƯỚC (base thuần) — làm mốc
python -m finetune.eval_mbpp --label before

# train QLoRA
python -m finetune.train_lora --epochs 3

# eval SAU (base + adapter)
python -m finetune.eval_mbpp --adapter finetune/adapters/deepseek-1.3b-base-lora --label after
```

Kết quả eval: `results/finetune/{before,after}_<ts>.json` với `total_pass1`.

## Ghi chú thiết kế

- **QLoRA, không full fine-tune:** RTX 3080 (10-12GB) đủ cho QLoRA (~4-6GB), KHÔNG đủ
  full FT 1.3B. Vài trăm mẫu → LoRA nhẹ, tránh overfit/catastrophic forgetting.
- **Teacher mạnh:** nếu teacher yếu (1.3b/6.7b) → thành "self-instruct", ghi rõ hạn chế.
- **Kỳ vọng thực tế:** cải thiện vài % (spec). Giá trị là "nhỏ nhưng đo được + giải
  thích sample complexity", không phải phép màu.
- **Adapter giữ riêng** (không merge) → bật/tắt để đối chứng before/after sạch.
