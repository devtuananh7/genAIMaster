# M5 — Fine-tune (OSS-Instruct thu nhỏ, QLoRA)

> **Trạng thái:** pipeline hoàn tất & tự kiểm thử (Stage A chạy thật với teacher mock); **số
> liệu train/eval chưa có** — Stage B/C chạy trên desktop RTX 3080. Xem `finetune/README.md`.

## 1. Bài toán là gì

M2/M3/M4 đều là **inference-time** — không đụng trọng số. M5 minh hoạ **trục training-time**:
**fine-tune** thực sự thay đổi trọng số model để phủ nốt nửa còn lại của bức tranh phân rã sai
số. Câu hỏi: với vài trăm mẫu dữ liệu tổng hợp, fine-tune 1.3b-**base** có nâng pass@1 không,
và nâng bao nhiêu?

⚠️ **Bẫy model:** M5 dùng deepseek-coder-1.3b-**BASE** (khác nhánh **instruct** của M1–M4). So
sánh **hợp lệ duy nhất**: base-TRƯỚC vs base-SAU fine-tune. **KHÔNG** so với instruct.

## 2. Cách giải là gì

Pipeline OSS-Instruct thu nhỏ, chia theo máy (`finetune/`):

```
   Stage A (data — không cần GPU)
     collect_seeds  → 343 seed 1-15 dòng từ boltons (BSD), ghi rõ nguồn
     gen_teacher    → teacher API bên thứ 3 sinh (problem, solution, tests) mỗi seed
     filter_runnable→ LỌC bằng Executor M0 (bỏ mẫu lời giải không chạy được)
     build_dataset  → train.jsonl (prompt style KHỚP eval)
   Stage B (train — RTX 3080, CUDA)
     train_lora     → QLoRA 4-bit + LoRA r=16 trên 1.3b-base, 2-3 epoch
   Stage C (eval — RTX 3080)
     eval_mbpp      → pass@1 before (base thuần) vs after (base+adapter), 50 MBPP của M0
```

- **Vì sao QLoRA, không full fine-tune:** RTX 3080 (10–12GB) đủ QLoRA (~4–6GB) nhưng **không
  đủ** full FT 1.3B (cần ~16–24GB cho weights+grad+Adam). Vài trăm mẫu → full FT overfit/quên;
  LoRA học một "delta" hạng thấp đắp lên base **đóng băng**. Adapter giữ riêng (không merge) →
  bật/tắt để đối chứng before/after sạch.
- **Teacher qua API bên thứ 3** (OpenAI-compatible): DeepSeek/OpenAI/Together/OpenRouter.

## 3. Dựa trên paper nào và phần nào

- **Paper nền: OSS-Instruct / Magicoder** — Wei et al., *"Magicoder: Source Code Is All You
  Need"* (2023).
- **Phần mượn:**
  - **Quy trình sinh dữ liệu:** rút **seed snippet 1–15 dòng** từ repo mã nguồn mở, đưa cho
    **Teacher LLM** (thường GPT-4) sinh cặp *(đặc tả bài toán NL, lời giải khép kín chạy được)*.
    M5 làm đúng khâu này, cộng thêm **lọc executability** bằng Executor M0.
  - **SFT tối ưu Cross-Entropy** trên tập instruction-code sinh ra (áp cho model < 7B).
- **Phần tự viết:** pipeline thu seed → sinh → lọc, script QLoRA train + eval.

## 4. Liên quan gì đến slide môn học

- **Week2 – Learning Theory (Generalization & Sample Complexity):** chất lượng model phụ thuộc
  **tính đại diện của tập huấn luyện** với đa tạp dữ liệu thật. OSS-Instruct **mở rộng H** và
  **giảm Approximation Error** bằng cách đa dạng hoá phân bố huấn luyện lấy từ code thực. Với
  chỉ vài trăm mẫu, **sample complexity** dự báo cải thiện **khiêm tốn** — đây là điểm phân tích
  chính (Week2 tr.33), không kỳ vọng phép màu.
- **Cross-Entropy / KL-divergence:** SFT tối ưu cross-entropy = giảm KL giữa phân bố dự đoán của
  model và phân bố dữ liệu → đúng khung tối ưu hoá của slide.
- **Đối chiếu hai trục sai số:** M5 (training-time, giảm Approximation) đặt cạnh M2/M3/M4
  (inference-time, giảm Estimation/tăng Robustness) cho báo cáo một bức tranh **phân rã sai số
  đầy đủ hai phía**.

## 5. Input/Output thực tế & nhận xét

**Input (1 dòng train.jsonl sau Stage A):**
```
### System: You are an expert Python programmer...
### User: Write a function `double_all(nums)`... It must pass these tests:
          assert double_all([1,2,3]) == [2,4,6]
### Assistant: ```python
def double_all(nums): return [n*2 for n in nums]
```   ← lời giải đã QUA LỌC executor (chạy đúng)
```

**Output kỳ vọng (Stage C):** hai file `results/finetune/{before,after}_<ts>.json` với
`total_pass1` — bảng so sánh base-1.3b **trước vs sau**.

**Nhận xét & trạng thái:**
- **Đã kiểm chứng trên Mac:** Stage A chạy trọn với teacher mock (343 seed → sinh → **lọc qua
  Executor M0** → train.jsonl đúng format). 7 unit test pass. File CUDA compile-clean (torch chỉ
  nạp trong `main()` nên import được cả trên máy không GPU).
- **Điểm cộng hệ thống:** khâu lọc **tái dùng Executor M0** — chỉ giữ mẫu lời giải *thực sự chạy
  được*, đảm bảo dữ liệu train sạch (đúng tinh thần "functional correctness" của cả POC).
- **Rủi ro cần lưu ý:** (1) teacher phải **đủ mạnh** — endpoint HF hiện tại (1.3b-instruct) quá
  yếu, sẽ biến thành "self-instruct" (ghi rõ hạn chế); nên dùng DeepSeek/GPT/Qwen-32B. (2) Kỳ
  vọng cải thiện **vài %** — giá trị nằm ở "nhỏ nhưng đo được + giải thích sample complexity".
- **M5 là stretch goal** (cắt đầu tiên nếu thiếu giờ), nhánh độc lập — nên để riêng, không chặn
  M1–M4.
