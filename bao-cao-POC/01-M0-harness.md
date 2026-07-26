# M0 — Experiment Harness (bộ khung thí nghiệm)

## 1. Bài toán là gì

Mọi kỹ thuật của POC (baseline, reflexion, RAG, multi-agent, fine-tune) cần được đo trên
**cùng một thước đo, cùng một tập bài, cùng một cách chấm** — nếu không, mọi so sánh về
sau đều vô nghĩa. Bài toán của M0: xây một **bộ khung dùng chung** để bất kỳ "chiến lược"
nào cũng *cắm vào chạy* 50 bài MBPP và **tự chấm pass@1**, xuất kết quả JSON + CSV.

M0 là **đường găng** (critical path): mọi module khác phụ thuộc nó (`blocks: M1–M5`).

## 2. Cách giải là gì

Một pipeline mỏng, không dùng thư viện agent (để giải thích được khi bảo vệ), gồm 6 mảnh:

```
   MBPP Loader ──► Strategy.solve(task) ──► Code Extractor ──► Sandbox Executor ──► Scorer
   (50 task đóng     (baseline/reflexion/     (bóc code khỏi      (chạy từng assert     (pass@1,
    băng)             rag/... cắm vào)         ```python fence)    trong subprocess)     JSON+CSV)
```

- **ExecutionResult schema** — "hợp đồng giao tiếp" M0 ↔ các module, chốt sớm nhất:
  `{status, stdout, stderr, traceback, failed_test, passed_count, total_count, duration_ms}`.
- **Sandbox Executor** (`harness/executor.py`) — thành phần rủi ro nhất: ghi `code + 1 assert`
  ra file tạm, chạy `subprocess.run([python, file], timeout=10)`. **Chạy từng assert riêng**
  để biết chính xác assert nào rớt đầu tiên (→ feedback cho M2/M4). Timeout 10s bắt buộc
  (code sinh ra có thể lặp vô hạn). Phân loại lỗi: `error_syntax / fail_assert / error_runtime
  / timeout / pass`.
- **Code Extractor** (`harness/extractor.py`) — nguồn "lỗi vặt" lớn nhất của loại POC này:
  ưu tiên khối ```python … ```, fallback về khối ``` bất kỳ, rồi về dòng `def` đầu tiên.
  **Có unit test riêng** (`tests/test_extractor.py`) — yêu cầu của Definition of Done.
- **Interface Strategy** — `class có .name và .solve(task) -> str`. Đủ để mọi module cắm vào.
- **Scorer/Runner** — chạy 50 bài qua 1 strategy, tính pass@1, ghi `results/<module>/<ts>.json`
  + CSV + `summary.json` (append mỗi lần chạy).

## 3. Dựa trên paper nào và phần nào

M0 không đề xuất thuật toán mới; nó **kế thừa quy ước đo lường** của hai benchmark chuẩn:

- **MBPP** — Austin et al., *"Program Synthesis with Large Language Models"* (Google
  Research, 2021). M0 dùng **phần benchmark & định dạng task** (mô tả NL + `test_list` các
  assert). Ta dùng biến thể **sanitized**.
- **HumanEval** — Chen et al., *"Evaluating Large Language Models Trained on Code"* (OpenAI,
  2021). M0 mượn **định nghĩa `pass@k`** và **cách chấm bằng thực thi** (functional
  correctness qua execution, thay vì so khớp chuỗi). Logic chấm tham khảo `openai/human-eval`.

Điểm cốt lõi kế thừa từ hai paper: **đo bằng *chạy code*, không đo bằng xác suất token** —
một hàm được coi là đúng chỉ khi vượt qua toàn bộ unit test.

## 4. Liên quan gì đến slide môn học

- **Week2 – Learning Theory (Empirical Risk):** pass@1 trên 50 bài chính là một ước lượng
  **rủi ro thực nghiệm** (empirical risk) của "hàm sinh code" trên một mẫu hữu hạn. 50 bài
  = tập kiểm tra cố định; con số pass@1 là ước lượng có phương sai (n nhỏ) — nền tảng để các
  module sau đo *delta* một cách nhất quán.
- **Định nghĩa hàm mục tiêu:** M0 hiện thực hoá việc tối ưu **functional correctness** (chạy
  đúng) thay vì **token-level likelihood** — đúng tinh thần "đo cái ta thực sự quan tâm" mà
  các phương pháp execution-feedback (Reflexion, CodeRL) nhấn mạnh.

## 5. Input/Output thực tế & nhận xét

**Input (1 task MBPP đã chuẩn hoá):**
```json
{ "task_id": 3,
  "text": "Write a python function to identify non-prime numbers.",
  "test_list": ["assert is_not_prime(2) == False", "assert is_not_prime(10) == True"],
  "test_imports": [] }
```

**Output (ExecutionResult khi chạy 1 đoạn code):**
```json
{ "status": "fail_assert", "failed_test": "assert is_not_prime(2) == False",
  "passed_count": 0, "total_count": 2, "duration_ms": 84, "traceback": "..." }
```

**Lệnh chạy & DoD đã đạt:**
```
python -m harness.run --strategy baseline --tasks data/selected_tasks.json
→ results/baseline/<ts>.json + .csv + summary.json
```

**Nhận xét:**
- Quyết định "**chạy từng assert riêng**" trả giá bằng tốc độ (mỗi assert 1 subprocess) nhưng
  đổi lại **feedback định vị được lỗi** — chính là nguyên liệu sống còn cho M2 (Reflexion) và
  M4 (Reviewer). Nếu gộp mọi assert vào 1 lần chạy, ta mất thông tin "assert nào rớt trước".
- **Extractor có test riêng** là quyết định đúng: khi soi lỗi baseline (M1), ta loại trừ được
  giả thuyết "lỗi do bóc code sai" ngay lập tức → khoanh vùng đúng nguyên nhân (code model).
- Về sau harness được nâng cấp: client sinh chuyển từ Ollama (LAN) sang **HuggingFace TGI
  endpoint**, và scorer thêm metadata per-vòng (`rounds_to_pass`, `internal_records`) để phục
  vụ đường cong pass@1-theo-vòng của M2/M4. Interface `solve()` giữ nguyên → không phá vỡ M1.
