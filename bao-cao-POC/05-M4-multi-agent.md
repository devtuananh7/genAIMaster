# M4 — Multi-Agent (Programmer ↔ Reviewer, ChatDev thu nhỏ)

## 1. Bài toán là gì

M2 (Reflexion) cho model **tự phê bình** chính mình. Câu hỏi M4: **phê bình từ một tác tử
Reviewer riêng biệt** có bắt lỗi tốt hơn tự phản tỉnh không? Tức so sánh **self-critique (M2)**
vs **external critique (M4)** trên cùng ngân sách vòng lặp.

## 2. Cách giải là gì

Hai tác tử giao tiếp trong vòng lặp (`multiagent/strategy.py`), tối đa 4 vòng (bằng M2 để công
bằng):

```
   Programmer sinh code ──► Execute (M0) ──► Reviewer nhận xét ──► Programmer sửa ──► lặp
        (đúng prompt baseline)              (đóng vai reviewer khó tính,
                                             KHÔNG được viết code hộ — ép trong prompt)
```

- **Reviewer** nhận `đề bài + code lỗi + ExecutionResult`, trả **nhận xét có cấu trúc**: *Error
  Location → Root Cause → Fix Direction*. **Ràng buộc cứng:** không viết code, chỉ nhận xét.
- **Cấu hình nâng cao:** cờ `reviewer_sees_execution` — nếu tắt, Reviewer phải tự suy lỗi chỉ
  bằng đọc code (tách bạch giá trị "đôi mắt thứ hai" vs "thông tin thực thi").
- Dùng lại ~70% hạ tầng vòng lặp M2 (chỉ thay bước Reflect bằng lượt gọi Reviewer).

## 3. Dựa trên paper nào và phần nào

- **Paper nền: ChatDev** — Qian et al., *"Communicative Agents for Software Development"* (2023);
  liên hệ **MetaGPT**.
- **Phần mượn:** khung **dual-agent communication** và tư tưởng **chia vai chuyên biệt** trong
  vòng đời phần mềm. ChatDev tổ chức bối cảnh qua **ChatChain** (phân rã công việc thành các nút
  ngữ cảnh độc lập, giới hạn nhiễu). M4 lấy phiên bản tối giản: chỉ 2 vai Programmer/Reviewer.
- **Phần tự viết:** Reviewer agent + tích hợp trên vòng lặp M2.

## 4. Liên quan gì đến slide môn học

- **Week2 – Learning Theory (No-Free-Lunch Theorem):** không tồn tại một model/thuật toán *phổ
  quát* giỏi mọi việc. Suy ra: một LLM khó vừa lập trình vừa tự soi lỗi hoàn hảo → **chia vai
  chuyên biệt** (Programmer sinh, Reviewer soi) là hệ quả trực tiếp của No-Free-Lunch. Đây là
  luận cứ lý thuyết cho kiến trúc đa tác tử.
- **Robustness / inference-time reasoning:** giống Reflexion, M4 biến sinh code tĩnh thành quá
  trình **tìm kiếm + tự hoàn thiện**, nhưng tín hiệu sửa đến từ **một tác tử khác** thay vì tự
  thân — kiểm nghiệm giả thuyết "góc nhìn thứ hai giảm điểm mù".

## 5. Input/Output thực tế & nhận xét

**Ví dụ Input/Output thật (task heap_queue_largest):**
```
Programmer → code (thiếu định nghĩa hàm đúng tên)
Execute    → NameError: 'heap_queue_largest' is not defined
Reviewer   → "ERROR LOCATION: hàm chưa được định nghĩa đúng tên...
              ROOT CAUSE: lệch tên entry-point... FIX DIRECTION: đặt đúng tên + dùng heapq.nlargest"
Programmer → code mới theo nhận xét
```

**Kết quả (số liệu thật, tập 150 bài, cùng backend HF):**

| | pass@1 | So baseline |
|--|--------|-------------|
| Baseline (M1) | 62.0% (93/150) | — |
| Reflexion (M2) | 70.0% (105/150) | +8.0đ |
| **Multi-Agent (M4)** | **70.7%** (106/150) | **+8.7đ** |

**Kiểm định McNemar ghép cặp:**
- baseline → M4: 17 bài fail→pass, 4 pass→fail, net **+13**, χ²=6.86, **p=0.0088 → có ý nghĩa**.
- M2 → M4: net +1, **p=1.0 → KHÔNG khác biệt** giữa self-critique và external-critique.

**Phân bố số vòng:** 99 bài pass vòng 1; **7 bài pass NHỜ review** (4 vòng 2, 1 vòng 3, 2 vòng 4);
46 bài chạy đủ 4 vòng vẫn fail. Toàn bộ I/O mỗi vòng lưu trong `internal_records`
(`programmer_prompt`, `raw_response`, `reviewer_prompt`, `reviewer_feedback`).

**Nhận xét (đã giải quyết caveat backend):**

- **M4 được "giải oan":** con số cũ **20%** (tập 50, backend Ollama) hoàn toàn do **backend
  lệch** — không phải do multi-agent kém. Sau khi đổi `ollama_client → hf_client` và chạy lại
  trên **cùng HF backend + cùng 150 bài**, M4 đạt **70.7%**, **thắng baseline có ý nghĩa**
  (p=0.0088). Bài học: *đối chứng phải cùng backend* — đúng mục tiêu số 1 của M4.
- **Trả lời câu hỏi nghiên cứu:** external critique (Reviewer riêng) **KHÔNG tốt hơn** self-
  critique (M2) với model 1.3b — hai bên **ngang nhau** (70.7% vs 70.0%, p=1.0). Cùng cứu được
  ~6–7 bài nhờ vòng lặp. Kết luận: *với model nhỏ trên MBPP, "người thứ hai" không thêm giá trị
  so với tự phản tỉnh* — một kết quả gọn và đáng bàn.
