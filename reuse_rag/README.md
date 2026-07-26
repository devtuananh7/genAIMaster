# reuse_rag — M3 (biến thể): API-grounded Reuse RAG

Trả lời câu hỏi: *cho một project Python cho sẵn, khi context cung cấp API có sẵn**
**giàu bao nhiêu thì model tái sử dụng đúng API đó nhiều bấy nhiêu?***

Đây là biến thể **reuse-first** của M3 (RepoCoder): thay vì đo *correctness* trên bài
xoá-thân-hàm, ta đo trực tiếp **reuse-rate** trên bài sinh-code-mới, và giữ retrieval
cố định (oracle) để cô lập biến **input representation**.

## Corpus & benchmark

- **Project corpus:** [`boltons`](https://github.com/mahmoud/boltons) — thư viện utility
  Python thật, pip-installable. Index bằng AST: `data/boltons_index.json` (236 symbol).
- **Benchmark:** `data/repo_tasks.json` — 12 task, mỗi task có `target_module`/`target_name`
  (API cần tái dùng) + `test_list` (unit test). API chọn có tên **không đoán được từ đề**
  để mức L1 khó reuse → đường cong nhúc nhích.

## 4 mức "input representation" (biến độc lập, lồng nhau, token tăng dần)

| Mức | Nội dung context |
|-----|------------------|
| L1 `signature`  | đường dẫn import + dòng chữ ký |
| L2 `+docstring` | L1 + mô tả (prose, đã bỏ doctest) |
| L3 `+example`   | L2 + một ví dụ gọi input→output |
| L4 `+body`      | L3 + toàn bộ mã nguồn hàm |

## Thành phần

```
reuse_rag/
  indexer.py        AST index package → chunk {fqn,sig,docstring,example,body}
  render.py         render_context(chunk, level) — 4 mức  ← biến độc lập
  reuse_scorer.py   AST: code có tái dùng đúng API? (chống dương tính giả)
  strategy.py       ReuseRagStrategy(level): oracle retrieve → render → generate
  tasks.py          RepoTask (kèm target API) ↔ harness.Task
  run_experiment.py driver: levels × tasks → results + đường cong
```

Tái dùng harness M0: `executor` (pass-rate), `extractor`, `hf_client`.

## Chạy

```bash
# 0. cài phụ thuộc (boltons là corpus)
./.venv/bin/pip install -r requirements.txt

# 1. build index (đã có sẵn data/boltons_index.json; chạy lại nếu cần)
./.venv/bin/python -m reuse_rag.indexer

# 2. CHẠY THẬT (cần token endpoint HF, giống M1/M2)
export HF_TOKEN=...
./.venv/bin/python -m reuse_rag.run_experiment --samples 3

# 2b. tự kiểm thử plumbing khi CHƯA có token (mock, KHÔNG phải số liệu thật)
./.venv/bin/python -m reuse_rag.run_experiment --mock
```

## Kết quả

Ghi vào `results/reuse_rag/`:
- `<ts>.json` — chi tiết per-level / per-task / per-sample (code, reuse, status).
- `curve_<ts>.csv` — đường cong: `level, reuse_rate, pass_rate, avg_context_tokens`.
- ASCII curve in ra stdout.

**Metric headline:** `reuse_rate` (chấm tĩnh bằng AST). `pass_rate` là lan can
correctness. `avg_context_tokens` cho trục chi phí → tìm "điểm ngọt" giàu/tốn.
