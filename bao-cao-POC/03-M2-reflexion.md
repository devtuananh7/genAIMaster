# M2 — Reflexion / Self-Debugging (tự gỡ lỗi qua phản tỉnh)

## 1. Bài toán là gì

Baseline (M1) sinh một lần rồi thôi — sai là sai. Nhưng khi chạy code, **trình biên dịch cho
tín hiệu đúng/sai rất giá trị** (traceback, assert nào rớt) mà baseline vứt đi. Bài toán M2:
cho model **tối đa N vòng tự đọc lỗi và sửa lại**, đo mức cải thiện pass@1 — mà **không cập
nhật trọng số**, chỉ tối ưu ở pha suy luận.

## 2. Cách giải là gì

Vòng lặp 3 pha **Act – Evaluate – Reflect** (`reflexion/strategy.py`), tối đa 4 vòng:

```
   [Generate] ── code ──► [Execute (Sandbox M0)] ──► [Check]
        ▲                                              │ pass → dừng, ghi số vòng
        │                                              │ fail
        │                                              ▼
   prompt = đề bài + episodic_memory ◄── [Reflect: sinh TỰ PHÊ ngắn ≤100 từ]
```

- **Vòng 0** sinh giống baseline (tái dùng prompt đã đóng băng).
- **Reflect:** model đọc `đề bài + code lỗi + feedback (loại lỗi + traceback ≤15 dòng + failed_test)
  + toàn bộ episodic_memory`, rồi viết một đoạn **tự phê** (root cause + phải đổi gì), **chưa
  sinh code**. Đoạn tự phê nối vào **episodic_memory** (chống lặp lại lỗi cũ).
- **Generate kế tiếp:** đề bài + toàn bộ episodic_memory → sinh code mới.
- Giới hạn: traceback ≤15 dòng, mỗi tự phê ≤100 từ (chống context phình sau nhiều vòng).

## 3. Dựa trên paper nào và phần nào

- **Paper nền: Reflexion** — Shinn et al., *"Reflexion: Language Agents with Verbal
  Reinforcement Learning"*, **arXiv 2303.11366 (NeurIPS 2023)**; repo `noahshinn/reflexion`.
- **Phần mượn:**
  - **Vòng lặp Act–Evaluate–Reflect** và **Episodic Memory** (mục kiến trúc tác tử của paper).
  - Ý tưởng **verbal reinforcement learning / "semantic gradient"**: chuyển tín hiệu nhị phân
    của compiler (chạy đúng/sai) thành **gradient bằng lời** — model đóng vai hàm phần thưởng
    tự nhiên. **Không cập nhật trọng số** (đúng thiết kế gốc): tối ưu hoá dời hẳn sang
    **in-context learning** ở pha suy luận.
- **Phần tự viết:** toàn bộ vòng lặp tích hợp trên harness M0 (executor làm khâu Evaluate).

## 4. Liên quan gì đến slide môn học

- **Week2 – Learning Theory (thoát tối ưu cục bộ & Estimation Error):** sinh một-đường-truyền
  (single trajectory) dễ kẹt ở **cực tiểu cục bộ** của quá trình giải mã. Reflexion biến sinh
  code tĩnh thành **bài toán tìm kiếm + tự hoàn thiện** (search & self-improvement) có định
  hướng bằng tín hiệu thực thi → giảm **Estimation Error** *ngay tại pha suy luận*, không đụng
  tới trọng số (không giảm Approximation Error).
- **"System 2 thinking":** thay vì ép trả lời đúng tức thì (System 1), model được trang bị
  vòng lặp phản tỉnh có chủ đích — liên hệ trực tiếp tới thảo luận robustness/inference-time
  reasoning trong tài liệu môn.

## 5. Input/Output thực tế & nhận xét

**Kết quả (số liệu thật, tập 150 bài, cùng backend HF):**

| | pass@1 | So baseline |
|--|--------|-------------|
| Baseline (M1) | 62.0% (93/150) | — |
| **Reflexion (≤4 vòng)** | **70.0%** (105/150) | **+8.0đ** |

**Kiểm định McNemar ghép cặp (cùng 150 bài):** 15 bài fail→pass, 3 bài pass→fail, net **+12**,
χ²=6.72, **p=0.0095 → có ý nghĩa thống kê** (ở tập 50 bài delta +8đ CHƯA significant — đây là
lý do mở rộng lên 150).

**Phân bố số vòng (`rounds_to_pass`):** 99 bài pass ngay vòng 1; **6 bài pass NHỜ phản tỉnh**
(4 bài ở vòng 2, 2 bài ở vòng 3); 45 bài chạy đủ 4 vòng vẫn không pass. Toàn bộ prompt+output
mỗi vòng lưu trong `internal_records` (`generate_prompt`, `raw_response`, `reflect_prompt`,
`self_critique`).

**Ví dụ luồng thật (rút gọn):**
```
Vòng 0  Generate → code
        Execute  → fail_assert: "assert ... == [85,75,65]"
Reflect → tự phê: "Hàm trả về sai thứ tự/độ dài; cần sort giảm dần rồi lấy k phần tử đầu."
Vòng 1  Generate (đề bài + tự phê) → code mới
        Execute  → pass ✓   (rounds_to_pass = 2)
```

**Nhận xét:**
- **+8 điểm là cải thiện thật và rẻ** (chỉ tốn thêm lời gọi ở pha suy luận, 0 chi phí train).
  Đúng loại lỗi mà M1 dự đoán feedback sẽ sửa được: **sai logic** (`fail_assert`) — code chạy
  được nhưng kết quả sai, phản tỉnh đọc được assert rớt và chỉnh.
- **Trần lợi ích ~vòng 3–4** đúng như paper: model 1.3b sau vài vòng bắt đầu **lặp lỗi cũ** hoặc
  làm context phình — lý do ta giới hạn traceback/tự phê và dùng episodic_memory chống lặp.
- **Cạm bẫy quan sát được:** thỉnh thoảng model "sửa" bằng cách hard-code output của test — cần
  ghi nhận thủ công như một điểm thảo luận (reflexion không phân biệt được "hiểu bài" vs "gian
  lận qua test").
- **Kết luận lý thuyết:** Reflexion giảm Estimation Error tại inference — nhưng **không mở rộng
  H** (không dạy model kiến thức mới). Đó là ranh giới mà M5 (fine-tune) mới vượt được.
