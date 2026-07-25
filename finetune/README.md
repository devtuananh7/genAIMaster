# M5 — Fine-tune với OSS-Instruct (LoRA)

Kỹ thuật **training-time** duy nhất trong project — tinh chỉnh `deepseek-coder-1.3b-base` bằng LoRA, sử dụng dữ liệu sinh theo phương pháp **OSS-Instruct**.

## Tổng quan Pipeline

```
   Repo mã nguồn mở          Teacher (qwen3.5:9b)         Sandbox Executor (M0)
         │                          │                             │
         ▼                          ▼                             ▼
┌─────────────────┐    ┌──────────────────────┐    ┌──────────────────────┐
│ 1. collect_seeds│───▶│ 2. generate_data     │───▶│ 3. filter_data       │
│   300-500 hàm   │    │   cặp (đề, lời giải) │    │   loại mẫu lỗi      │
│   ngắn (1-15 L) │    │   từ mỗi seed        │    │   reuse M0 executor  │
└─────────────────┘    └──────────────────────┘    └──────────────────────┘
                                                            │
                                                            ▼
                                                   ┌──────────────────┐
                                                   │ 4. format_training│
                                                   │   JSONL cho LoRA  │
                                                   └──────────────────┘
                                                            │
                                                            ▼
                                                   ┌──────────────────┐
                                                   │ 5. train_lora    │
                                                   │   Mac: mlx_lm    │
                                                   │   PC:  peft/QLoRA│
                                                   └──────────────────┘
                                                            │
                                                            ▼
                                                   ┌──────────────────┐
                                                   │ 6. evaluate      │
                                                   │   TRƯỚC vs SAU   │
                                                   │   50 bài MBPP    │
                                                   │   N=3 runs       │
                                                   └──────────────────┘
```

## Thiết lập

### Máy Mac Apple Silicon M4 (16GB RAM)

```bash
cd /path/to/genAIMaster
pip install -r finetune/requirements_mac.txt
```

### PC RTX 4060 (8GB VRAM, 16GB RAM)

```bash
cd /path/to/genAIMaster
pip install -r finetune/requirements_pc.txt
```

## Chạy từng bước

### Step 1: Thu thập seed snippets

```bash
# Clone 3 repo OSS + trích xuất 300-500 hàm ngắn
python -m finetune.collect_seeds

# Tuỳ chỉnh:
python -m finetune.collect_seeds --target 400 --max-lines 12
```

Output: `finetune/data/seed_snippets.json`

### Step 2: Sinh dữ liệu bằng giáo viên (qwen3.5:9b)

```bash
# Chạy đầy đủ (mất ~1-2h tuỳ tốc độ model)
python -m finetune.generate_data

# Chạy thử 10 mẫu trước:
python -m finetune.generate_data --limit 10

# Tiếp tục nếu bị gián đoạn:
python -m finetune.generate_data --resume
```

> **Lưu ý:** Cần Ollama server với model `qwen3.5:9b`. 
> Đặt biến `OLLAMA_HOST` nếu server không phải localhost.

Output: `finetune/data/raw_pairs.json`

### Step 3: Lọc dữ liệu

```bash
python -m finetune.filter_data
```

Output: `finetune/data/filtered_pairs.json` + `filter_stats.json`

### Step 4: Format dữ liệu

```bash
python -m finetune.format_training
```

Output: `finetune/data/train.jsonl`, `finetune/data/valid.jsonl`

### Step 5: Train LoRA

```bash
# Mac M4 (mlx_lm):
python -m finetune.train_lora --backend mlx

# PC RTX 4060 (peft QLoRA):
python -m finetune.train_lora --backend peft

# Tuỳ chỉnh:
python -m finetune.train_lora --backend mlx --epochs 3 --lr 2e-4 --batch-size 4
```

Output: `finetune/adapters/mlx/` hoặc `finetune/adapters/peft/`

### Step 6: Đánh giá

```bash
# Mac:
python -m finetune.evaluate --backend mlx --runs 3

# PC:
python -m finetune.evaluate --backend transformers --runs 3

# Chỉ đánh giá TRƯỚC fine-tune (chưa có adapter):
python -m finetune.evaluate --backend mlx --adapter none --runs 1
```

Output: `finetune/results/eval_<timestamp>.json`

## Model

| Variant | Model | Mục đích |
|---|---|---|
| TRƯỚC fine-tune | `deepseek-coder-1.3b-base` (vanilla) | Baseline nhánh training-time |
| SAU fine-tune | `deepseek-coder-1.3b-base` + LoRA adapter | So sánh cải thiện |

> ⚠️ **KHÔNG** so sánh với model instruct (M1–M4). Hai nhánh tách biệt.

## Cấu hình LoRA

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| Rank (r) | 16 | |
| Alpha (α) | 32 | α/r = 2 |
| Dropout | 0.05 | |
| Target modules | `q_proj`, `v_proj` | Attention projections |
| Epochs | 3 | |
| Learning rate | 2×10⁻⁴ | Cosine decay |
| Batch size | 4 | Mac: mlx. PC: QLoRA 4-bit |

## Kế thừa vs Tự viết (inheritance)

| Kế thừa | Tự viết |
|---|---|
| Ý tưởng OSS-Instruct (MagicCoder paper) | Pipeline thu seed + sinh + lọc |
| `peft` / `mlx_lm` cho LoRA | Script train, evaluate |
| `harness.executor` (M0) cho lọc dữ liệu | Strategy cho evaluation |
| Ollama API cho teacher model | Prompt engineering giáo viên |

## Kỳ vọng

Với vài trăm mẫu huấn luyện, kỳ vọng cải thiện **khiêm tốn** (vài điểm %).
Kết quả "cải thiện nhỏ nhưng đo được" + phân tích vì sao (sample complexity)
vẫn là một chương báo cáo có giá trị.
