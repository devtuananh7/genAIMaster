# TÀI LIỆU BACKBONE — POC "TỐI ƯU HÓA LLM SINH MÃ NGUỒN" (chuẩn 1.3b)

**Môn học:** IT5410 – Nền tảng AI Tạo sinh (Foundation of Generative AI)
**Chủ đề BTL:** Chủ đề gợi ý số 3 (Week0, trang 11) — *"Lập trình và phát triển phần mềm: Gợi ý mã lập trình, hoàn thành đoạn code, tạo hàm mới từ ngữ cảnh, hoặc tạo test case tự động"*
**Nhóm:** 5 thành viên · **Ngày lập:** 14/07/2026 · **Chuẩn:** 1.3b (thay bản gốc 6.7b)

> **Cách đọc bộ tài liệu này:**
> - File `.md` này là **bản kể chuyện** (cho người, cho báo cáo): bài toán, lý thuyết, timeline.
> - Chi tiết thi công cho **agent/coder** nằm trong các file YAML — mỗi module một file:
>   - `00-contract.yaml` ⭐ — **hợp đồng đóng băng** (model, tham số, schema, 50 task). Đọc TRƯỚC.
>   - `01-M0-harness.yaml` · `02-M1-baseline.yaml` · `03-M2-reflexion.yaml`
>   - `04-M3-rag.yaml` · `05-M4-multiagent.yaml` · `06-M5-finetune.yaml` · `07-M6-report.yaml`
> - Bản gốc 6.7b được lưu lịch sử tại `draft/`.

**Các quyết định công nghệ đã chốt:**

| Hạng mục | Quyết định |
|---|---|
| Ngôn ngữ | Python thuần (không dùng framework agent như LangChain) |
| LLM chạy local | **`deepseek-coder:1.3b` (instruct)** qua Ollama tại **`http://192.168.31.16:11434`** — mọi thành viên dùng CÙNG model |
| ⚠️ Thay đổi so với gốc | Bản gốc dùng 6.7b; **hạ chuẩn xuống 1.3b do máy chủ không đủ RAM/VRAM**. Đây là *deviation* bắt buộc ghi rõ trong báo cáo (Week0 tr.10) |
| Benchmark | MBPP — Mostly Basic Python Problems, subset ~50 bài |
| Embedding cho RAG | `nomic-embed-text` qua Ollama |
| Fine-tune | LoRA trên `deepseek-coder-1.3b-base` bằng `mlx_lm` (Mac) hoặc Google Colab |
| Quỹ thời gian | ~4 tuần kể từ 14/07 |

---

# PHẦN 1 — TỔNG QUAN CÁC VẤN ĐỀ CẦN LÀM

## 1.1. Phát biểu bài toán (Step 1, Week0 tr.8: "Problem should be well defined")

> **Bài toán:** Cho một đề bài lập trình mô tả bằng ngôn ngữ tự nhiên kèm unit test, hệ thống phải sinh ra một hàm Python **chạy đúng** — vượt qua toàn bộ unit test.
>
> - **Input:** chuỗi mô tả yêu cầu, vd *"Write a function to find the shared elements from the given two lists"* + 3 unit test của MBPP.
> - **Output:** mã nguồn Python của hàm.
> - **Thước đo:** `pass@1` — % số bài mà code sinh ở lượt đầu vượt qua toàn bộ unit test.

## 1.2. Câu hỏi nghiên cứu của POC

LLM cỡ nhỏ chạy local (**deepseek-coder 1.3b**) sinh code sai khá nhiều. **Các kỹ thuật can thiệp ở pha suy luận (inference-time) và pha huấn luyện (training-time) cải thiện được bao nhiêu?** POC trả lời bằng thí nghiệm ablation:

```
[A] Baseline ──► [B] +RAG ──► [C] +Reflexion ──► [D] +Multi-Agent      [E] Fine-tune
 sinh 1 lần      thêm ngữ      vòng lặp tự        Reviewer riêng        (nhánh độc lập,
 rồi chấm        cảnh repo     gỡ lỗi             thay tự phản tỉnh     so với [A])
```

> **Điểm lợi bất ngờ của model yếu:** câu chuyện của POC là *delta* (kỹ thuật cải thiện được bao nhiêu), không phải giá trị tuyệt đối. Baseline 1.3b thấp (~40-50%) để lại **headroom lớn** → đường cong cải thiện của Reflexion/RAG/Multi-Agent dễ ấn tượng hơn so với khi chạy 6.7b (vốn đã gần trần). Biến ràng buộc phần cứng thành một luận điểm thảo luận.

## 1.3. Kiến trúc tổng thể

```
┌─────────────────────────────────────────────────────────────────┐
│                      EXPERIMENT HARNESS (M0)                     │
│              (bộ khung thí nghiệm dùng chung cho cả nhóm)        │
│                                                                  │
│   MBPP Loader ──► Prompt Builder ──► Ollama Client ──► Code      │
│   (nạp 50 bài)    (dựng prompt)      192.168.31.16     Extractor │
│        │                             :11434 (1.3b)        │      │
│        │              ┌───────────────────────────────────┘      │
│        ▼              ▼                                          │
│   Sandbox Executor (subprocess cách ly, timeout 10s,             │
│   bắt stdout/stderr/traceback) ──► ExecutionResult (hợp đồng)    │
│        │                                                         │
│        ▼                                                         │
│   Scorer (chấm pass@1) ──► results/*.json ──► bảng ablation      │
└─────────────────────────────────────────────────────────────────┘
     ▲            ▲              ▲               ▲
  M1 Baseline  M2 Reflexion   M3 RAG        M4 Multi-Agent
                                                          M5 Fine-tune (nhánh riêng)
```

## 1.4. Sản phẩm bàn giao (Week0 tr.9)

1. **Mã nguồn** GitHub + notebook Colab minh họa.
2. **Readme** hướng dẫn cài đặt và chạy.
3. **Báo cáo LaTeX** (Overleaf) — bám khung *Input representation / Training / Prediction* (Week0 tr.8, Step 2).
4. **Slides** thuyết trình.

**Tiêu chí chấm (Week0 tr.10):** *"Nếu sử dụng lại/kế thừa/khai thác các mã nguồn/gói/công cụ sẵn có thì phải nêu rõ ràng và chính xác trong báo cáo"* — mỗi module ghi rõ phần kế thừa vs tự viết (xem trường `inheritance` trong từng YAML).

---

# PHẦN 2 — CƠ SỞ LÝ THUYẾT: LIÊN KẾT VỚI SLIDE MÔN HỌC

Phần này là "xương sống học thuật" của báo cáo. Mỗi kỹ thuật được neo vào khái niệm trong slide, **trích dẫn nguyên văn kèm số trang**. Bảng tổng hợp trích dẫn dạng máy đọc nằm trong `07-M6-report.yaml#theory_citations`.

## 2.1. Nền chung: bài toán học máy và phân rã sai số

**Slide gốc: `Week2-Learning Theory.pdf`**

- **Hypothesis space** — tr.10: *"Hypothesis space (model space): a set ℋ of functions, providing candidates h for a learning algorithm"*. LLM sinh code là một hàm h trong ℋ khổng lồ; mọi kỹ thuật POC là cách "khoanh vùng" hoặc "dò tìm tốt hơn" trong ℋ.
- **ERM** — tr.14: *"In practice, a learner cannot access F(P,h). It should rely on F(D,h)… However, h_erm may get overfitting"*. Điểm xuất phát để lập luận **vì sao huấn luyện thông thường chưa đủ cho sinh code**.
- **Error decomposition** — tr.16–17: *"Error(h_o) := Optimization error + Generalization error + Approximation error"*. **Khung phân loại 7 kỹ thuật**:
  - OSS-Instruct → *Approximation error*.
  - RL + Compiler Feedback → *Generalization error*.
  - Reflexion/RAG/Multi-Agent → giảm sai số **tại pha suy luận, không đổi trọng số**.
- **Data manifold** — tr.18. **Bias-Variance** — tr.22. **No-free-lunch theorem** — tr.34: *"No universal learner!"* → cơ sở lý thuyết của Multi-Agent.

## 2.2. Nền chung: mô hình tự hồi quy và MLE

**Slide gốc: `Week6-autoregressive-models.pdf` và `Week5-VAE.pdf`**

- **Chain rule & autoregressive** — Week6 tr.8: *"…**You can choose your own order of the variables.**"* — nền tảng lý thuyết của FIM.
- **MLE** — Week5 tr.24: *"θ* = argmax (1/m) Σ log p_θ(x)"*.
- **KL-divergence** — Week5 tr.21–22: *"Learning by minimizing KL(P‖P_θ)…"*. SFT trong OSS-Instruct = tối thiểu KL.

## 2.3. Kỹ thuật 1 — FIM (Fill-in-the-Middle)

Tái cấu trúc dữ liệu thành `<PRE> Prefix <SUF> Suffix <MID> Middle`. **Liên kết:** Week4 tr.47–48 (Masked Multi-Head Attention — causal mask giữ nguyên), Week6 tr.8. **Vai trò POC:** không train FIM; M3 (RAG) tận dụng chế độ FIM có sẵn của deepseek-coder.

## 2.4. Kỹ thuật 2 — Repository-Level Semantic RAG

Embed các đoạn code, truy xuất top-k đưa vào prompt; RepoCoder thêm vòng lặp truy xuất–sinh. **Liên kết:** Week4 tr.55 (*"Quadratic compute in self-attention"* — O(n²) là rào cản buộc chọn lọc ngữ cảnh), Week8 tr.53 (CLIP contrastive — retriever huấn luyện y hệt, thay cặp (ảnh, text) bằng (query, code)).

## 2.5. Kỹ thuật 3 — RL with Compiler Feedback

Dùng kết quả biên dịch/test làm reward. StepCoder thêm CCCS + FGO. **Liên kết:** Week2 tr.14 (giới hạn ERM), Week6 tr.14–15 (MADE masking — FGO cùng triết lý dùng mask kiểm soát luồng tín hiệu học). Bài giảng RL của môn (tuần sau) là chỗ dựa trực tiếp — cập nhật trích dẫn khi có slide.

## 2.6. Kỹ thuật 4 — Self-Debugging & Reflexion

Vòng lặp *Generate → Execute → Reflect*, không cập nhật trọng số (verbal reinforcement learning). **Liên kết:** Week2 tr.16–17 — Reflexion đặc biệt ở chỗ **giảm sai số mà không đổi hypothesis space** — luận điểm phân tích trung tâm của báo cáo.

## 2.7. Kỹ thuật 5 — TDD Harness & Kỹ thuật 6 — OSS-Instruct

**TDD Harness:** code sinh ra bắt buộc qua Verifier mới được trả. **Liên kết:** Week8 tr.36/44/76 (Classifier / Classifier-Free Guidance / Dynamic thresholding), Week6 tr.19 (Conditional models — unit test là "điều kiện c").
**OSS-Instruct:** bốc 1–15 dòng code thật làm seed, LLM giáo viên sáng tác (đề, lời giải) → SFT. **Liên kết:** Week5 tr.21–22,24 (KL & MLE), Week2 tr.18 (data manifold), tr.22 (bias-variance).

## 2.8. Kỹ thuật 7 — Multi-Agent Collaboration

Nhiều LLM đóng vai chuyên biệt (Programmer, Reviewer, Designer…). **Liên kết:** Week2 tr.34 — No-free-lunch theorem: không model nào giỏi mọi vai → phân rã thành xã hội tác tử chuyên môn hóa.

---

# PHẦN 3 — CHI TIẾT TỪNG MODULE

> Chi tiết đầy đủ (components, steps, DoD, traps, inheritance) nằm trong các file YAML tương ứng. Dưới đây là tóm tắt điều hướng.

| Module | File YAML | Vai trò | Chủ | Kỳ vọng |
|---|---|---|---|---|
| **M0** Harness | `01-M0-harness.yaml` | ⭐ Đường găng — mọi module gọi qua | A | xong 21/7 |
| **M1** Baseline | `02-M1-baseline.yaml` | Mốc so sánh gốc | A | pass@1 ~40-50% |
| **M2** Reflexion | `03-M2-reflexion.yaml` | Tự gỡ lỗi ≤4 vòng | B | đường cong iteration |
| **M3** RAG | `04-M3-rag.yaml` | Truy xuất repo (khó nhất) | C | pass-rate RepoTasks |
| **M4** Multi-Agent | `05-M4-multiagent.yaml` | Reviewer riêng vs tự phê | D | so cạnh M2 |
| **M5** Fine-tune | `06-M5-finetune.yaml` | Training-time (stretch) | E | cải thiện khiêm tốn |
| **M6** Báo cáo | `07-M6-report.yaml` | LaTeX + Slides (xuyên suốt) | E | — |

**Quy ước chung:** tham số sinh cố định (`temperature=0.2`, `max_tokens=1024`); biến thể lấy mẫu dùng `temperature=0.8` và phải ghi rõ. Mỗi run ghi: model, tham số, timestamp, danh sách bài, kết quả từng bài, tổng pass@1 (xem `00-contract.yaml#results_convention`).

**⭐ M0 là đường găng** — B, C, D, E đều chờ M0. Tuần 1 họ làm việc "không cần harness": đọc kỹ thuật của mình, viết mục lý thuyết, dựng benchmark/dữ liệu, chốt interface.

---

# PHẦN 4 — PHÂN CHIA CÔNG VIỆC (5 THÀNH VIÊN)

## 4.1. Phân vai

| Vai | Module | Ghi chú chọn người |
|---|---|---|
| **A — Trục kỹ thuật** | M0 + M1, rồi quản repo, tổng hợp bảng ablation | Code Python vững nhất — cả nhóm bị chặn bởi M0 |
| **B** | M2 Reflexion + ablation memory/feedback | Cẩn thận prompt engineering |
| **C** | M3 RAG (nặng nhất) | Mạnh thứ hai của nhóm |
| **D** | M4 Multi-Agent (phối hợp chặt với B) | Bắt cặp với B tuần 1 chốt interface |
| **E** | M5 Fine-tune + khung LaTeX | Cần máy khỏe hoặc quen Colab |

**Nguyên tắc:** mỗi người tự viết phần cơ sở lý thuyết + kết quả module mình trong báo cáo.

## 4.2. Timeline 4 tuần

```
            Tuần 1 (15-21/7)    Tuần 2 (22-28/7)    Tuần 3 (29/7-4/8)   Tuần 4 (5-11/8)
           ┌──────────────────┬───────────────────┬───────────────────┬────────────────┐
A: M0+M1   │██████ M0 harness │█ M1 baseline+lỗi  │ hỗ trợ + tổng hợp │ bảng ablation  │
B: M2      │ đọc lý thuyết,   │████ vòng lặp      │██ ablation memory │ viết báo cáo   │
           │ chốt interface   │     Reflexion     │                   │                │
C: M3      │██ RepoTasks      │████ indexer +     │███ vòng lặp       │ viết báo cáo   │
           │   benchmark      │     retriever     │    RepoCoder      │                │
D: M4      │ chốt interface   │████ Reviewer      │██ so sánh với M2  │ viết báo cáo   │
           │ với B            │     agent         │                   │                │
E: M5+M6   │██ skeleton LaTeX │██ seed + sinh     │███ LoRA + đánh    │ slides + tổng  │
           │   + thu seed     │   dữ liệu         │    giá            │ duyệt          │
           └──────────────────┴───────────────────┴───────────────────┴────────────────┘
 Mốc chốt:  M0 xong 21/7 ⭐    Mọi module chạy     Khóa số liệu 4/8    Nộp + thuyết trình
            (đường găng)       được bản đầu 28/7
```

**Phụ thuộc:** B/C/D/E cần M0 → tuần 1 làm việc không cần harness. M4 phụ thuộc interface vòng lặp M2 → B và D chốt chung tuần 1. Bảng ablation cần M1–M4 khóa số liệu trước 4/8; M5 được phép trễ.

## 4.3. Cơ chế phối hợp

- **Sync 2 lần/tuần** (30 phút): trạng thái module, blocker, thay đổi schema.
- **Quy tắc vàng** (xem `00-contract.yaml#golden_rules`): không ai đổi `ExecutionResult` schema, 50 task, prompt baseline, hay tham số sinh mà không báo cả nhóm — 4 thứ giữ kết quả 5 người so được với nhau.
- **Thứ tự hy sinh khi trễ:** M5 (bỏ hẳn) → ablation phụ M2 → vòng 2 M3. **Không bao giờ cắt:** M0, M1, và ít nhất một trong M2/M4.

---

*Tài liệu backbone (chuẩn 1.3b). Chi tiết thi công theo từng file YAML kèm theo. Chi tiết task từng ngày sẽ cụ thể hóa trong OpenSpec change proposal khi triển khai.*
