# M1 — Baseline (mốc so sánh gốc)

## 1. Bài toán là gì

Trước khi nói "kỹ thuật X cải thiện được bao nhiêu", ta cần **con số gốc**: deepseek-coder
1.3b sinh code **một lần duy nhất**, không kỹ thuật hỗ trợ, đạt pass@1 bao nhiêu trên 50 bài
MBPP? Đây là **mốc neo** mà M2/M3/M4 so vào, và **prompt baseline** ở đây bị **đóng băng** để
mọi module sau kế thừa nguyên văn (đảm bảo so sánh công bằng).

Code rất ít; **giá trị nằm ở phân tích lỗi thủ công** — "mỏ vàng" cho chương thảo luận.

## 2. Cách giải là gì

- **Prompt baseline** (`baseline/strategy.py`): đề bài MBPP + **chữ ký hàm gợi ý** (parse tên
  hàm + tham số từ assert đầu tiên) + yêu cầu "chỉ trả code trong khối ```python". Sinh 1 lần,
  `temperature=0.2` (greedy-ish), 1 mẫu/bài.
- **Phân tích lỗi thủ công (bước quan trọng nhất):** phân loại 50 kết quả về 5 nhóm: *sai cú
  pháp, lệch tên/hiểu sai đề, sai logic, lỗi format, timeout*.
- **Biến thể sampling** (`temperature=0.8`, 5 mẫu/bài): kiểm tra "bắn nhiều mẫu có cứu được
  model yếu không?".

## 3. Dựa trên paper nào và phần nào

M1 không dựa trên một paper thuật toán — nó là **điểm tham chiếu (reference point)**. Hai liên
hệ học thuật:
- Kế thừa **định nghĩa pass@k** và giao thức đánh giá của **HumanEval** (Chen et al. 2021) và
  **MBPP** (Austin et al. 2021) — sinh trực tiếp từ mô tả bài toán, chấm bằng thực thi.
- Biến thể sampling liên hệ trực tiếp tới thảo luận **pass@k vs pass@1**: tăng số mẫu k là một
  dạng *tìm kiếm ngẫu nhiên* trong không gian đầu ra — M1 kiểm nghiệm xem nó có hiệu quả không.

## 4. Liên quan gì đến slide môn học

- **Week2 – Learning Theory (Hypothesis Space & năng lực xấp xỉ):** Baseline chính là chất
  lượng của **hàm giả thuyết thô** mà model rút ra từ không gian giả thuyết `H` của nó dưới
  giải mã tham lam. Sinh đúng ngay lần đầu (zero-shot) là bài toán khó — bị chặn bởi **giới
  hạn năng lực xấp xỉ** của model 1.3b. Baseline thấp = `H` của model nhỏ chưa phủ tốt đa tạp
  lời giải → tạo **headroom** cho các can thiệp sau.
- **Bias–variance & sampling:** biến thể temp cao 5 mẫu = tăng phương sai đầu ra để mong "bắn
  trúng". Kết quả (xem dưới) cho thấy với model yếu, tăng phương sai **không** giảm được sai số
  hệ thống — cần can thiệp có cấu trúc, khớp lời cảnh báo của lý thuyết.

## 5. Input/Output thực tế & nhận xét

**Con số chính (số liệu thật):**

| Cấu hình | pass@1 |
|----------|--------|
| **Baseline chốt** (có chữ ký hàm, temp 0.2, HF endpoint) | **52%** (26/50) |
| *Ablation:* KHÔNG cấp chữ ký hàm (bản đầu, Ollama) | 22% (11/50) |
| Sampling 1 mẫu (temp 0.8) | 12% |
| Sampling best-of-5 (temp 0.8) | 22% |

**Phân loại lỗi (bản 22%, greedy — trước khi thêm chữ ký hàm):**

| Nhóm lỗi | Số bài | % |
|----------|--------|---|
| ✓ pass | 11 | 22% |
| Sai cú pháp (`error_syntax`, `IndentationError`) | 21 | 42% |
| Lệch tên / hiểu sai đề (`NameError`) | 8 | 16% |
| Sai logic (`fail_assert`, `TypeError`) | 10 | 20% |
| Lỗi format đầu ra | 0 | 0% |

**Ví dụ Input/Output thật (task 3):**
```
Input  : "Write a python function to identify non-prime numbers."
         test: assert is_not_prime(2) == False
Output : def all_primes(n): ...        ← model đặt SAI tên hàm (khác test)
Chấm   : NameError: 'is_not_prime' is not defined  → rớt
```

**Hai nhận xét cho báo cáo:**

1. **Lấy mẫu ngẫu nhiên vô ích với model yếu:**
   ```
      greedy 1 mẫu (temp 0.2)  : 22%  ┐
      sampling 1 mẫu (temp 0.8): 12%  │  5 lần bắn temp cao chỉ vừa GỠ HOÀ
      sampling 5 mẫu (temp 0.8): 22%  ┘  với 1 lần greedy → lợi ích ròng = 0
   ```
   → Động lực trung tâm của POC: cần **can thiệp có cấu trúc** (Reflexion đọc lỗi rồi sửa, RAG
   thêm ngữ cảnh) chứ không phải brute-force resampling.

2. **Cấp chữ ký hàm là đòn bẩy rẻ mà mạnh:** 42% lỗi cú pháp + 16% lệch tên phần lớn do model
   1.3b không suy ra đúng entry-point. Thêm chữ ký hàm vào prompt đưa baseline **22% → 52%**
   — bản thân đây là một ablation nhỏ giá trị ("tác dụng của việc cấp chữ ký hàm"), và giải
   thích vì sao ta chốt 52% làm mốc chính cho M2–M4.
