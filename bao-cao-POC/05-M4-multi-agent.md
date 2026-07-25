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

**Kết quả (số liệu thật, `results/multiagent/`):**

| | pass@1 |
|--|--------|
| Baseline (M1) | 52% |
| Reflexion (M2) | 60% |
| **Multi-Agent (M4)** | **20%** |

**Nhận xét (kèm caveat quan trọng — cần đọc kỹ):**

- **Con số 20% THẤP hơn cả baseline** — bất thường. Có **hai lý do xếp chồng**, cần tách bạch
  trước khi kết luận:
  1. **Số liệu cũ / backend lệch:** run M4 thực hiện *trước* đợt nâng cấp, còn dùng backend
     **Ollama** trong khi baseline/M2 đã chuyển sang **HuggingFace endpoint**. Khác backend →
     **so sánh chưa công bằng** (đây là mục tiêu số 1 của M4 lại đang bị vi phạm).
  2. **Giả thuyết thật:** nhận xét dài của "reviewer khó tính" có thể **làm rối model 1.3b** →
     sinh code lệch (nhiều `NameError` — code không định nghĩa đúng hàm). Nếu đúng, đây là kết
     luận nghiên cứu thú vị: *external critique phản tác dụng với model nhỏ*.
- **Chưa phân biệt được (1) hay (2)** nếu chưa **re-run trên cùng HF backend** — đó chính là
  việc còn tồn đọng của M4.
- **Trạng thái module:** code M4 đã có đầy đủ tracking metadata; việc cần làm là đổi
  `ollama_client → hf_client` và chạy lại 50 task để có số liệu **công bằng** đặt cạnh đường
  cong M2. Chừng nào chưa re-run, **con số 20% chỉ nên trình bày kèm caveat**, không dùng để
  kết luận "multi-agent thua reflexion".
