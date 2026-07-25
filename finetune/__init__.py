"""
finetune — Module M5: Fine-tune (OSS-Instruct thu nhỏ, QLoRA) trên 1.3b-BASE
============================================================================
Trục TRAINING-TIME để đối chiếu với các kỹ thuật inference-time (M1-M4).

QUAN TRỌNG: M5 dùng deepseek-coder-1.3b-**BASE** (khác nhánh instruct của M1-M4).
So sánh DUY NHẤT hợp lệ: base-TRƯỚC vs base-SAU fine-tune. KHÔNG so với instruct.

Pipeline chia theo MÁY:
  Stage A (data)  — chạy ở ĐÂU CŨNG ĐƯỢC (Mac/desktop): thu seed → teacher API → lọc
  Stage B (train) — chạy trên DESKTOP RTX 3080 (CUDA): QLoRA fine-tune
  Stage C (eval)  — chạy trên DESKTOP: pass@1 before vs after trên 50 MBPP của M0
"""
