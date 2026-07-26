# Báo cáo POC — Tối ưu LLM sinh code tốt hơn (deepseek-coder:1.3b)

> Bộ tài liệu này là **báo cáo POC** hoàn chỉnh. Mỗi module M0–M5 là một tài liệu riêng,
> cùng một sườn: **(1) Bài toán — (2) Cách giải — (3) Paper nền & phần nào — (4) Liên hệ
> slide môn học — (5) Input/Output thực tế & nhận xét.**

## 1. Bối cảnh & câu hỏi nghiên cứu

Một mô hình sinh code **nhỏ** (deepseek-coder **1.3B**) sinh code sai khá nhiều khi chạy
một lần (zero-shot). Câu hỏi trung tâm của POC:

> Với **cùng một model nhỏ cố định**, các kỹ thuật *inference-time* (tự gỡ lỗi, truy xuất
> ngữ cảnh, đa tác tử) và *training-time* (fine-tune) **cải thiện chất lượng sinh code**
> đến đâu, và **vì sao** (soi qua lăng kính lý thuyết học máy)?

Ta cố định model, cố định benchmark (50 bài MBPP), chỉ thay đổi **chiến lược** — nhờ vậy
mọi so sánh là công bằng và quy được về một trục.

## 2. Khung lý thuyết xuyên suốt — Phân rã sai số (Week2 – Learning Theory)

Tổng sai số của một mô hình học:

```
   Total Error = Approximation Error  +  Estimation Error
                 (H đủ giàu chưa?)       (Optimization + Generalization)
```

Mỗi module tấn công một thành phần khác nhau — đây là "xương sống" gắn kết cả báo cáo:

| Module | Trục | Tấn công sai số nào |
|--------|------|---------------------|
| M1 Baseline   | — | Đo mốc gốc (raw hypothesis từ H) |
| M2 Reflexion  | inference-time | Estimation: thoát tối ưu cục bộ bằng vòng lặp tự sửa |
| M3 RAG        | inference-time | Approximation: bơm ngữ cảnh repo để bớt "bịa" API |
| M4 Multi-Agent| inference-time | Robustness: phê bình ngoài (No-Free-Lunch → chuyên biệt hoá) |
| M5 Fine-tune  | training-time | Approximation: mở rộng H bằng dữ liệu tổng hợp (OSS-Instruct) |

## 3. Kết quả tổng hợp (số liệu thật)

**Kết quả chốt trên tập 150 bài MBPP (cùng backend HF, McNemar ghép cặp):**

| Module | Kỹ thuật | Metric chính | Kết quả (150 bài) |
|--------|----------|--------------|---------|
| **M0** | Harness (executor + MBPP loader + scorer) | hạ tầng | tập 50 & 150 bài đóng băng, pass@1 tự động |
| **M1** | Baseline (1 lần sinh) | pass@1 | **62.0%** (93/150) — *22% nếu KHÔNG cấp chữ ký hàm* |
| **M2** | Reflexion (tự gỡ lỗi ≤4 vòng) | pass@1 | **70.0%** (105/150) — **+8.0đ**, McNemar p=0.0095 ✓ |
| **M3** | Reuse RAG (API-grounded, 4 mức context) | reuse-rate | **0% → 75%** (no-RAG → có API); pass 92%→≤83% |
| **M4** | Multi-Agent (Programmer↔Reviewer) | pass@1 | **70.7%** (106/150) — **+8.7đ**, p=0.0088 ✓; ngang M2 (p=1.0) |
| **M5** | QLoRA fine-tune 1.3b-base | pass@1 before/after | *pipeline sẵn sàng, chưa có số liệu (chạy trên RTX 3080)* |

> Tập 50 bài (bản đầu): baseline 52% · M2 60% · M4 20% (backend ollama, đã lỗi thời). Tập
> **150 bài** là bản chốt: delta +8đ **có ý nghĩa thống kê** (p<0.01) — ở 50 bài thì chưa.
> M4 sau khi sửa về **cùng backend HF** đạt 70.7%, **ngang M2** (không khác biệt, p=1.0).

**Hai phát hiện xuyên suốt:**
1. **Brute-force resampling vô ích với model yếu** (M1): bắn 5 mẫu temp cao chỉ hoà với 1
   lần greedy → phải **can thiệp có cấu trúc**, không phải lấy mẫu nhiều.
2. **Reuse ≠ Correctness** (M3): RAG là điều kiện *cần* để model tái dùng API (0→75%),
   nhưng ép reuse có thể *đánh đổi* độ đúng ở model 1.3b.

## 4. Cấu hình chung (đóng băng — `00-contract.yaml`)

- **Model:** deepseek-coder:1.3b-instruct (M1–M4); 1.3b-**base** (M5, tách riêng).
- **Sinh:** `temperature=0.2`, `max_tokens=1024`; biến thể sampling `temperature=0.8`.
- **Benchmark:** 50 bài MBPP sanitized, đóng băng trong `data/selected_tasks.json`.
- **Chấm:** pass@1 qua Sandbox Executor (subprocess, timeout 10s, chạy từng assert riêng).

## 5. Danh mục tài liệu

| File | Nội dung |
|------|----------|
| `01-M0-harness.md` | Bộ khung thí nghiệm (đường găng) |
| `02-M1-baseline.md` | Mốc gốc + phân tích lỗi thủ công |
| `03-M2-reflexion.md` | Tự gỡ lỗi / phản tỉnh |
| `04-M3-rag-reuse.md` | Truy xuất ngữ cảnh & tái dùng API |
| `05-M4-multi-agent.md` | Programmer ↔ Reviewer |
| `06-M5-finetune.md` | QLoRA fine-tune (OSS-Instruct thu nhỏ) |

## 6. Danh mục paper nền

| Kỹ thuật | Paper | arXiv / nguồn |
|----------|-------|---------------|
| Benchmark | MBPP (Austin et al. 2021); HumanEval (Chen et al. 2021) | google-research/mbpp; openai/human-eval |
| Reflexion | Shinn et al. 2023, *Language Agents with Verbal Reinforcement Learning* | arXiv 2303.11366 (NeurIPS 2023) |
| RepoCoder | Zhang et al. 2023, *Repository-Level Code Completion Through Iterative Retrieval and Generation* | arXiv 2303.12570 (EMNLP 2023) |
| Multi-Agent | ChatDev (Qian et al. 2023); MetaGPT | ChatDev: Communicative Agents for Software Dev |
| Fine-tune | Magicoder / OSS-Instruct (Wei et al. 2023) | *Source Code Is All You Need* |

Slide môn học liên quan: **Week2 – Learning Theory** (phân rã sai số, hypothesis space,
No-Free-Lunch, sample complexity), **Week4 – Transformer** (độ phức tạp self-attention →
giới hạn context → động lực cho RAG).
