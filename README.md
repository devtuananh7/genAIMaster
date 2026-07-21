# POC — Tối ưu hóa LLM sinh mã nguồn · Baseline (M0 + M1)

Bộ khung thí nghiệm (**M0 — Experiment Harness**) và mốc so sánh gốc (**M1 — Baseline**) cho đồ án *"Tối ưu hóa LLM sinh mã nguồn"* (môn IT5410 — Nền tảng AI Tạo sinh).

Mục tiêu POC: đo xem các kỹ thuật can thiệp ở pha suy luận (RAG, Reflexion, Multi-Agent) và pha huấn luyện (fine-tune) cải thiện được bao nhiêu cho một LLM sinh code cỡ nhỏ chạy local. Repo này chứa **nền móng** mà mọi module sau cắm vào.

---

## Bài toán

Cho một đề bài lập trình mô tả bằng ngôn ngữ tự nhiên + các unit test, hệ thống phải sinh ra một hàm Python **chạy đúng** (vượt toàn bộ test).

- **Benchmark:** MBPP (Mostly Basic Python Problems) — subset 50 bài đóng băng trong `data/selected_tasks.json`.
- **Thước đo:** `pass@1` — tỷ lệ bài mà code sinh ở lượt đầu vượt toàn bộ unit test.
- **Model:** `deepseek-coder:1.3b` (instruct) chạy qua [Ollama](https://ollama.com).

## Kiến trúc M0

```
MBPP Loader ──► Baseline Strategy ──► Ollama Client ──► Code Extractor
(50 task)       (dựng prompt)         (gọi model)       (tách ```python)
     │                                                        │
     └────────────────────────► Sandbox Executor ◄────────────┘
                                 (subprocess, timeout 10s)
                                        │
                                        ▼
                                 Scorer + Runner ──► results/<strategy>/<run>.json
```

| Thành phần | File | Vai trò |
|---|---|---|
| MBPP Loader | `harness/loader.py` | Nạp MBPP sanitized, chọn & đóng băng 50 task |
| Ollama Client | `harness/ollama_client.py` | Gọi `POST /api/chat`, retry + seed cố định |
| Code Extractor | `harness/extractor.py` | Tách code khỏi văn xuôi của model |
| Sandbox Executor | `harness/executor.py` | Chạy code + test trong tiến trình cách ly, timeout 10s |
| Scorer + Runner | `harness/scorer.py`, `harness/run.py` | Chấm pass@1, xuất JSON |
| Baseline Strategy | `baseline/strategy.py` | Sinh code 1 lần (M1) |
| Signature parser | `harness/signature.py` | Suy tên hàm entry-point từ assert |

Điểm mở rộng: mỗi module sau chỉ cần cung cấp một `Strategy` (`solve(task) -> code`) là cắm vào Runner được, không phải sửa M0.

## Cài đặt

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt   # datasets, requests, pytest
```

Cần một server Ollama đã nạp `deepseek-coder:1.3b`:
```bash
ollama pull deepseek-coder:1.3b
```
Mặc định harness trỏ tới `http://192.168.31.16:11434`; đổi qua biến môi trường `OLLAMA_HOST` hoặc cờ `--base-url`.

## Chạy

```bash
# (tùy chọn) chốt lại danh sách 50 task
.venv/bin/python -m harness.loader --output data/selected_tasks.json

# chạy baseline
.venv/bin/python -m harness.run --strategy baseline --tasks data/selected_tasks.json

# xem pass@1
.venv/bin/python -c "import json,glob; f=sorted(glob.glob('results/baseline/*.json'))[-1]; d=json.load(open(f)); print(f'pass@1 = {d[\"total_pass1\"]:.1%}')"
```

Biến thể lấy mẫu (pass@k tham chiếu):
```bash
.venv/bin/python -m harness.run --strategy baseline --tasks data/selected_tasks.json --temperature 0.8 --samples 5
```

## Kết quả baseline (deepseek-coder:1.3b)

**pass@1 ≈ 23%** (± 2). Phân bố lỗi trên 50 bài:

| Nhóm | % | Ghi chú |
|---|---|---|
| ✓ pass | ~22% | |
| sai cú pháp | ~42% | ngoặc lệch, `//` làm comment, one-liner hỏng |
| lệch tên / hiểu sai đề | ~16% | `NameError` — đã giảm nhờ ép tên hàm vào prompt |
| sai logic | ~20% | code chạy nhưng kết quả sai |

Chi tiết: `results/baseline/error_analysis.md` (sinh sau khi chạy).

### Hai bài học đáng ghi

1. **Tái lập được là bắt buộc.** Không cố định `seed`, model ở `temperature=0.2` vẫn ngẫu nhiên hoàn toàn — 2 lần chạy ra kết quả khác hẳn, mọi so sánh ablation vô nghĩa. Harness cố định `seed=5410`; phần dao động còn lại (~±2 điểm) là non-determinism tầng GPU, nên **mọi số ablation chính thức chạy N lần, báo mean ± std**.
2. **Lỗi model nhỏ xếp lớp.** Sửa một loại lỗi (tên hàm) chỉ làm lộ loại kế tiếp (cú pháp/logic) → tinh chỉnh single-shot cho lợi ích nhỏ. Đây chính là động lực cho các kỹ thuật lặp (Reflexion, Multi-Agent) ở module sau.

## Cấu trúc repo

```
harness/       # M0 — bộ khung dùng chung
baseline/      # M1 — strategy sinh code 1 lần
tests/         # unit test cho extractor
data/          # 50 task MBPP đã đóng băng
00-contract.yaml .. 07-M6-report.yaml   # spec đóng băng + kế hoạch từng module
backbone.md    # tài liệu tổng quan POC
```

`00-contract.yaml` là **hợp đồng đóng băng** (model, tham số sinh, schema `ExecutionResult`, 50 task) — không đổi nếu không thông báo cả nhóm, để kết quả các module so sánh được với nhau.

## Trạng thái

- [x] M0 — Experiment Harness
- [x] M1 — Baseline + phân tích lỗi
- [ ] M2 — Reflexion · [ ] M3 — RAG · [ ] M4 — Multi-Agent · [ ] M5 — Fine-tune
