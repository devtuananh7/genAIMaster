# M3 — RAG & Tái sử dụng API (API-grounded Reuse RAG)

> M3 hiện thực theo hướng **reuse-first**: biến thể của RepoCoder tập trung đo *trực tiếp* việc
> tái dùng API. Xem thêm `reuse_rag/BAO-CAO-HIEN-TRANG-M3.md` cho bản đầy đủ.

## 1. Bài toán là gì

Code thực tế không phải file đơn lẻ — lập trình viên **gọi API, dùng hàm đã định nghĩa ở file
khác**. Model nhỏ khi không có ngữ cảnh repo thường **bịa lại (reimplement)** hàm đã có thay vì
tái dùng. Bài toán M3:

> Cho một project Python cho sẵn, khi prompt yêu cầu một việc mà project **đã có API** để làm,
> **truy xuất đúng API và đưa vào ngữ cảnh** có giúp model **tái sử dụng** thay vì viết lại
> không? Và **"input representation" giàu bao nhiêu** thì reuse thay đổi thế nào?

## 2. Cách giải là gì

Một **RAG pipeline** (retrieval là bước cố định trong code — *không phải agent*). Corpus =
thư viện thật **boltons** (BSD), index bằng **AST** (236 symbol). Cô lập biến bằng **oracle
retrieval** (nạp thẳng API đích) để chỉ thay đổi **độ giàu context** qua 5 mức:

```
   L0 no_api     : KHÔNG đưa API (baseline no-RAG, đối chứng)
   L1 signature  : đường import + chữ ký
   L2 +docstring : L1 + mô tả
   L3 +example   : L2 + ví dụ gọi input→output
   L4 +body      : L3 + toàn bộ mã nguồn hàm
```

- **Metric headline: reuse-rate** — chấm tĩnh bằng AST (`reuse_scorer.py`): code có *import +
  gọi* đúng API đích không, **chống dương tính giả** (model tự định nghĩa lại tên → không tính).
- **pass-rate** (executor M0) làm lan can correctness; **token** làm trục chi phí.

## 3. Dựa trên paper nào và phần nào

- **Paper nền: RepoCoder** — Zhang et al., *"Repository-Level Code Completion Through Iterative
  Retrieval and Generation"*, **arXiv 2303.12570 (EMNLP 2023)**.
- **Phần mượn:** ý tưởng **truy xuất ngữ cảnh repo để model TÁI DÙNG hàm có sẵn thay vì tạo
  lại** — chính là luận điểm RepoCoder chứng minh giảm "code smell / technical debt" khi model
  bịa lại hàm. RepoCoder gốc dùng **vòng lặp truy xuất lặp** (draft-as-query); M3 giữ *tinh
  thần tái dùng* nhưng **đổi khung đo**: thay vì đo Exact-Match trên bài xoá-thân-hàm, ta đo
  **reuse-rate trực tiếp** trên bài sinh-mới → sạch và dễ chấm hơn cho model 1.3b.
- Liên hệ họ **InlineCoder / Context Inlining** (arXiv 2601.00376): nội tuyến định nghĩa API
  vào prompt — chính là điều các mức L1–L4 làm.
- **Phần tự viết:** toàn bộ AST indexer, renderer 5 mức, reuse-scorer, pipeline.

## 4. Liên quan gì đến slide môn học

- **Week4 – Transformer (độ phức tạp Self-Attention `O(n²·d)`):** không thể nhồi cả repo triệu
  dòng vào context window vì chi phí bậc hai theo độ dài chuỗi. RAG là **giải pháp trực tiếp**
  cho giới hạn kiến trúc này — chỉ truy xuất và nội tuyến phần *liên quan*. Đây là cầu nối chặt
  nhất giữa POC và slide Transformer.
- **Week2 – Learning Theory (Approximation Error):** cung cấp API đúng trong ngữ cảnh giúp model
  **bớt bịa** — giảm sai số do "không biết hàm tồn tại". RAG bù đắp cho `H` hữu hạn của model
  bằng tri thức ngoài (external knowledge) tại pha suy luận.

## 5. Input/Output thực tế & nhận xét

**Ví dụ Input/Output thật (task `chunked`, mức L3):**
```
Input  : "Viết split_into_chunks(items, size)..." + context L3:
         # Project API: boltons.iterutils.chunked
         def chunked(src, size, count=None): ...
         # Example: >>> chunked(range(10), 3)  →  [[0,1,2],[3,4,5],...]
Output : from boltons.iterutils import chunked
         def split_into_chunks(items, size):
             return [list(chunked(items, size)) for _ in range(len(items)//size)]
Chấm   : reuse=True (gọi đúng chunked)  ·  pass=False (bọc sai logic thừa)
```

**Kết quả (số liệu thật, deepseek-1.3b, 12 task × 3 sample):**

| Mức | reuse-rate | pass-rate | ~token |
|-----|-----------|-----------|--------|
| L0 no-API (baseline) | **0.0%** | 91.7% | 0 |
| L1 signature | **75.0%** | 83.3% | 15 |
| L2 +docstring | 75.0% | 66.7% | 43 |
| L3 +example | 50.0% | 41.7% | 51 |
| L4 +body | 66.7% | 83.3% | 121 |

**Nhận xét (3 phát hiện):**

1. **RAG là điều kiện *cần* cho reuse — thắng baseline áp đảo:** L0 = **0%** (không đưa API,
   model *không bao giờ* tự tái dùng boltons). Chỉ cần signature (L1) → **75%**.
2. **Reuse ≠ Correctness:** baseline pass 91.7% > RAG (42–83%). Ép reuse **đánh đổi** độ đúng:
   model 1.3b gọi đúng API nhưng bọc sai logic, hoặc **echo/derail** khi context dài. Vì các
   task này *đủ dễ để model tự giải*, reuse đo được nhưng **chưa load-bearing**.
3. **Đường cong KHÔNG đơn điệu, trũng ở L3:** context càng dài, model nhỏ càng dễ rối (echo cả
   dòng doctest `>>>` → lỗi cú pháp). **Signature tối giản là biểu diễn tốt nhất** để induce
   reuse ở 1.3b — một kết quả non-obvious.

**Caveat:** N=12 (sai số rộng); phần tụt L3 một phần là artifact của định dạng doctest thô.
Hướng tiếp: nâng độ khó benchmark (để reuse *cần thiết*), ví dụ dạng prose thay `>>>`.
