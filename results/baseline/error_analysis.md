# Phân tích lỗi Baseline M1 — deepseek-coder:1.3b (instruct)

**Ngày chạy:** 2026-07-21 · **Model:** `deepseek-coder:1.3b` (Q4_0, instruct) qua Ollama `192.168.31.16:11434`
**File nguồn:** `results/baseline/20260721T160400Z.json` (greedy), `20260721T161008Z.json` (sampling)

## Con số chính

| Run | Tham số | Kết quả |
|---|---|---|
| Greedy | `temp=0.2`, 1 mẫu/bài | **pass@1 = 22%** (11/50) |
| Sampling — 1 mẫu | `temp=0.8`, mẫu đầu | pass@1 = 12% (6/50) |
| Sampling — best of 5 | `temp=0.8`, 5 mẫu/bài | **pass@5 = 22%** (11/50) |

> Baseline 22% **thấp hơn kỳ vọng 40–50%** (contract `sanity_note`). Đã điều tra: extractor hoạt động đúng; nguyên nhân nằm ở code model sinh ra + prompt thiếu chữ ký hàm.

## Phân loại 50 kết quả greedy (temp 0.2)

Ánh xạ về 5 nhóm lỗi chuẩn của M1 (spec `baseline-evaluation`):

| Nhóm lỗi | Số bài | % | Nguồn kỹ thuật |
|---|---|---|---|
| ✓ **pass** | 11 | 22% | — |
| **sai cú pháp** | 21 | 42% | `error_syntax` (16) + `IndentationError` (5) |
| **lệch tên / hiểu sai đề** | 8 | 16% | `NameError` (8) — tên hàm/biến khác test |
| **sai logic** | 10 | 20% | `fail_assert` (8) + `TypeError` (1) + `PatternError` (1) |
| **lỗi format đầu ra** | 0 | 0% | extractor tách code sạch, không phát sinh |
| **timeout** | 0 | 0% | — |

**Danh sách task pass (greedy):** 4, 223, 224, 272, 277, 413, 447, 452, 562, 772, 798

## Đặc điểm lỗi (soi code thật)

**Sai cú pháp (42%) — model 1.3b viết code hỏng:**
- Ngoặc lệch trong one-liner dài (task 92, 111): nhồi list-comprehension + comment vào một dòng, không đóng nổi ngoặc.
- Dùng `//` làm comment kiểu C++/Java (task 119) → Python báo SyntaxError. Model nhầm cú pháp ngôn ngữ.
- Comment văn xuôi lê thê chèn giữa biểu thức làm gãy cú pháp.

**Lệch tên (16%) — NameError:** tasks 3, 106, 135, 256, 419, 435, 455, 751.
- Task 3: test gọi `is_not_prime(...)` nhưng model `def all_primes(...)` + gọi helper chưa định nghĩa.
- Task 106: test dùng `tuple1` nhưng model đặt tên tham số khác.
- **Nguyên nhân gốc:** prompt KHÔNG cấp chữ ký hàm (quyết định `design.md#D7`), model 1.3b không suy ra đúng tên entry-point từ assert.

**Sai logic (20%):** `fail_assert` — code chạy được nhưng kết quả sai; đây là "yếu thật" của model, đúng loại lỗi mà Reflexion/Multi-Agent kỳ vọng sửa được.

## Hai phát hiện cho báo cáo

**1. Lấy mẫu ngẫu nhiên vô ích với model yếu.**
```
   greedy  1 mẫu  temp 0.2 : 22%  ┐
   sampling 1 mẫu temp 0.8 : 12%  │  5 lần bắn temp cao chỉ vừa gỡ hòa
   sampling 5 mẫu temp 0.8 : 22%  ┘  với 1 lần greedy → 0 lợi ích ròng
```
→ Cần **can thiệp có cấu trúc** (Reflexion đọc lỗi rồi sửa, RAG thêm ngữ cảnh) chứ không phải brute-force resampling. Đây là động lực trung tâm của POC.

**2. Giả định `design.md#D7` bị phủ nhận.** "Model suy ra chữ ký hàm từ test" KHÔNG đúng với 1.3b (16% NameError). → Quyết định đảo: cấp chữ ký hàm trong prompt (chuẩn MBPP). Xem `design.md#D7`.

## Hành động tiếp theo

- [ ] Sửa prompt baseline: parse tên hàm + tham số từ assert đầu, đưa vào prompt → chạy lại (kỳ vọng ~30–45%).
- [ ] Giữ con số 22% "trần trụi" như một ablation nhỏ: "tác dụng của việc cấp chữ ký hàm".
