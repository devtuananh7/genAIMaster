# Báo cáo hiện trạng — M3 (biến thể): API-grounded Reuse RAG

- **Ngày:** 2026-07-25
- **Module:** `reuse_rag/` — biến thể reuse-first của M3 (thay cho RepoCoder infilling)
- **Model:** deepseek-coder:1.3b (HuggingFace TGI endpoint, giống M1/M2)
- **Corpus:** [boltons](https://github.com/mahmoud/boltons) — thư viện utility Python thật
- **Trạng thái:** pipeline hoàn tất, đã chạy số liệu thật (baseline L0 + 4 mức L1–L4)

---

## 1. Bài toán & luận điểm

Cho một project Python cho sẵn, khi prompt yêu cầu làm một việc mà project **đã có API**
để làm, model có **tái sử dụng đúng API** đó thay vì viết lại không? Và **"input
representation" giàu bao nhiêu** thì reuse thay đổi thế nào?

Đây là biến thể **reuse-first** của M3: thay vì đo *correctness* trên bài xoá-thân-hàm
(RepoCoder), ta đo trực tiếp **reuse-rate** trên bài sinh-code-mới, giữ retrieval cố
định (oracle — nạp thẳng API đích) để **cô lập biến input representation**.

Đây là **RAG pipeline** (retrieval là bước cố định trong code), **không phải agent**
(model không tự quyết định đi tra cứu). Lựa chọn đúng cho model nhỏ 1.3b.

---

## 2. Thiết kế thí nghiệm

**Benchmark:** 12 RepoTask (`data/repo_tasks.json`), mỗi task gồm prompt mô tả hành vi,
một API đích trong boltons cần tái dùng (tên **không đoán được từ đề**), và unit test.

**Index:** AST index toàn package boltons → `data/boltons_index.json` (**236 symbol**),
mỗi chunk `{fqn, signature, docstring, example, body}`.

**Biến độc lập — 5 mức context (lồng nhau, token tăng dần):**

| Mức | Nội dung context |
|-----|------------------|
| **L0** `no_api`    | **không đưa API nào** — baseline no-RAG (đối chứng) |
| L1 `signature`     | đường dẫn import + dòng chữ ký |
| L2 `+docstring`    | L1 + mô tả (prose, bỏ doctest) |
| L3 `+example`      | L2 + một ví dụ gọi input→output |
| L4 `+body`         | L3 + toàn bộ mã nguồn hàm |

**Metric:**
- `reuse_rate` — **headline**, chấm tĩnh bằng AST (có chống dương tính giả: model tự
  định nghĩa lại tên API không tính là reuse).
- `pass_rate` — lan can correctness (chạy unit test qua executor M0).
- `avg_context_tokens` — trục chi phí.

**Cấu hình:** temperature 0.2, max_tokens 1024, **3 sample/task**, seed cố định.

---

## 3. Kết quả (số liệu thật)

```
                  reuse-rate      pass-rate     ~token
   ────────────────────────────────────────────────────
   L0 no-API  ◄baseline   0.0%       91.7%        0
   L1 signature          75.0%       83.3%       15
   L2 +docstring         75.0%       66.7%       43
   L3 +example           50.0%       41.7%       51
   L4 +body              66.7%       83.3%      121
```

**Ma trận reuse per-task (0/1, trung bình 3 sample):**

| task (API đích)      | L0 | L1 | L2 | L3 | L4 |
|----------------------|----|----|----|----|----|
| chunked              | 0  | 1  | 1  | 1  | 1  |
| windowed             | 0  | 1  | 1  | 1  | 1  |
| first                | 0  | 0  | 0  | 0  | 1  |
| bucketize            | 0  | 1  | 1  | 1  | 0  |
| unique               | 0  | 1  | 1  | 1  | 1  |
| pairwise             | 0  | 1  | 1  | 1  | 1  |
| flatten              | 0  | 1  | 1  | 0  | 0  |
| slugify              | 0  | 1  | 1  | 1  | 1  |
| ordinalize           | 0  | 1  | 1  | 0  | 1  |
| cardinalize          | 0  | 0  | 0  | 0  | 1  |
| bytes2human          | 0  | 1  | 1  | 0  | 0  |
| clamp                | 0  | 0  | 0  | 0  | 0  |

Nguồn: `results/reuse_rag/20260725T222524Z.json` (L0), `20260725T221755Z.json` (L1–L4).

---

## 4. Phân tích

### 4.1. Trên trục REUSE: RAG hơn baseline áp đảo ✅
`0% → 75%`. Khi **không** đưa API (L0), model **không bao giờ** tự tái dùng boltons
(0/12). Chỉ cần đưa signature (L1) là reuse vọt lên 75%. **RAG là điều kiện cần để
model tái dùng code project** — đây là luận điểm cốt lõi, được chứng minh sạch.

### 4.2. Trên trục CORRECTNESS: RAG thua baseline ❌
Baseline pass `91.7%` vs RAG `42–83%`. Đưa API context **làm giảm** độ đúng của model
1.3b. Hai cơ chế quan sát được:
- **Derail / echo context:** model sao chép lại context (kể cả dòng doctest `>>>`) vào
  output thay vì viết hàm mới → `error_syntax`. Xảy ra nhiều ở L3/L4 (context dài).
- **Reuse đúng nhưng bọc sai logic:** ví dụ task `chunked`, model gọi đúng
  `chunked(items, size)` nhưng thêm list-comprehension thừa → sai kết quả.

### 4.3. Đường cong KHÔNG đơn điệu — trũng ở L3
Ngược với giả thuyết "giàu hơn → reuse nhiều hơn". Reuse **cao nhất ở L1** (chỉ
signature), **tụt đáy ở L3** (50%), hồi một phần ở L4. Context càng dài, model nhỏ càng
dễ rối. **Signature tối giản là biểu diễn hiệu quả nhất** để induce reuse ở model 1.3b.

### 4.4. Giới hạn quan trọng của benchmark
`baseline pass 91.7%` ⇒ 12 task này **dễ tới mức model tự giải được, không cần API**.
Nghĩa là reuse **đang đo được nhưng chưa load-bearing** (chưa quyết định đúng/sai). Để
reuse trở nên *cần thiết*, benchmark cần các task mà model **không tự giải nổi** (hàm
phức tạp, nhiều bước). Khi đó "RAG → reuse → pass tăng" mới thành câu chuyện trọn vẹn.

---

## 5. Kết luận

> RAG **cần thiết và hiệu quả cho tái sử dụng** (0%→75%; model không bao giờ tự reuse).
> Nhưng với model 1.3b, ép reuse bằng context **đánh đổi bằng correctness** (92%→≤83%),
> và context càng dài càng hại (đáy ở L3). Signature tối giản là biểu diễn tốt nhất.
> Benchmark nên nâng độ khó để reuse trở nên load-bearing.

**Caveat:** N=12 task (0.75 = 9/12) → sai số rộng; đừng over-claim độ dốc. Phần tụt ở L3
một phần là artifact của định dạng ví dụ doctest thô `>>>` (dễ bị echo).

---

## 6. Hướng tiếp theo (đề xuất, chưa làm)

1. **Nâng độ khó benchmark** — chọn API mà 1.3b tự giải không nổi → reuse load-bearing.
2. **Ví dụ dạng prose** thay doctest `>>>` → tách "context dài có hại" khỏi "doctest gây rối".
3. **Làm bền extractor** — loại phần context bị echo để giảm `error_syntax`.
4. **Thí nghiệm RETRIEVAL** (đang oracle) — thêm cosine top-k trên index để đo recall.

---

## 7. Tái lập

```bash
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python -m reuse_rag.indexer                 # build data/boltons_index.json
export HF_TOKEN=...                                     # endpoint HF
./.venv/bin/python -m reuse_rag.run_experiment --levels 0,1,2,3,4 --samples 3
./.venv/bin/python -m pytest tests/test_reuse_rag.py -q # 10 test
```

Chi tiết pipeline: xem `reuse_rag/README.md`.
