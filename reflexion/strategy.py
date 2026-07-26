"""
reflexion/strategy.py
=====================
Mô-đun M2 — Reflexion / Self-Debugging
---------------------------------------
Nguồn gốc ý tưởng: paper "Reflexion" (Shinn et al., 2023) + repo noahshinn/reflexion
Phần tự viết     : vòng lặp Act–Evaluate–Reflect tích hợp harness M0

Thuật toán 3 pha (theo 03-M2-reflexion.yaml):
  [Generate] → [Execute] → [Check]
                              ↓ fail
                          [Reflect]  → episodic_memory → [Generate] (lặp lại)

Điểm quan trọng:
  - Vòng 0: sinh code giống hệt Baseline (không tốn thêm lời gọi AI).
  - Pha Reflect: sinh ra một đoạn TỰ PHÊ ngắn (≤100 từ), CHƯA sinh lại code.
  - Vòng Generate tiếp theo mới sinh code mới dựa trên đề bài + toàn bộ episodic_memory.
  - Traceback được rút gọn ≤15 dòng để chống context phình (theo contract M0).
"""

from __future__ import annotations

from baseline.strategy import BASE_SYSTEM_PROMPT  # Prompt cơ bản dùng chung với M1
from harness.executor import run as run_executor   # Sandbox Executor của M0
from harness.extractor import extract_code         # Bóc tách code Python khỏi văn xuôi AI
from harness.hf_client import generate             # Client gọi HuggingFace Inference Endpoint
from harness.types import Task                     # Kiểu dữ liệu bài toán MBPP

# ---------------------------------------------------------------------------
# CÁC HẰNG SỐ (theo 03-M2-reflexion.yaml và 00-contract.yaml)
# ---------------------------------------------------------------------------
MAX_TRACEBACK_LINES = 15   # Giới hạn dòng traceback tối đa gửi cho AI (chống context phình)
MAX_REFLECT_WORDS = 100    # Giới hạn số từ mỗi đoạn tự phê trong episodic_memory

# Prompt hệ thống cho Pha Reflect:
# Yêu cầu AI PHÂN TÍCH lỗi và đưa ra tự phê ngắn gọn (CHƯA sinh code mới)
REFLECT_SYSTEM_PROMPT = """You are an expert Python programmer performing self-reflection.
Analyze the failed code and its error, then write a SHORT self-critique (under 100 words).
Focus on: root cause of the error + what to change next.
Do NOT write any code yet."""

# Prompt hệ thống cho Pha Generate (sửa code):
# Yêu cầu AI đọc toàn bộ lịch sử tự phê và viết lại code chính xác
REGENERATE_SYSTEM_PROMPT = """You are an expert Python programmer.
You have analyzed your previous mistakes and written self-critiques.
Now write a corrected Python function that passes all unit tests.
Return ONLY the executable Python code block in ```python ... ``` fences."""


def _shorten_traceback(stderr: str) -> str:
    """
    Rút gọn traceback xuống tối đa MAX_TRACEBACK_LINES dòng cuối cùng.
    Tránh prompt phình quá lớn sau nhiều vòng lặp (context window 4096 token).
    """
    lines = stderr.strip().splitlines()
    if len(lines) <= MAX_TRACEBACK_LINES:
        return "\n".join(lines)
    # Chỉ giữ lại N dòng cuối quan trọng nhất (thường là dòng lỗi thực sự)
    return "...(truncated)...\n" + "\n".join(lines[-MAX_TRACEBACK_LINES:])


def _build_feedback(result) -> str:
    """
    Tổng hợp thông tin phản hồi từ ExecutionResult thành đoạn văn bản rõ ràng cho AI.
    Ưu tiên theo thứ tự: failed_test > traceback > stderr > status.
    Theo spec 03-M2-reflexion.yaml: 'loại lỗi + traceback rút gọn + failed_test'
    """
    parts = []

    # Loại lỗi tổng quát
    parts.append(f"Error type: {result.status}")

    # Câu lệnh assert cụ thể bị rớt (CỰC KỲ QUAN TRỌNG cho model nhỏ 1.3b)
    if getattr(result, "failed_test", None):
        parts.append(f"Failed test: {result.failed_test}")

    # Traceback rút gọn (nếu có)
    if getattr(result, "traceback", None):
        parts.append(f"Traceback:\n{_shorten_traceback(result.traceback)}")
    elif getattr(result, "stderr", None):
        parts.append(f"Stderr:\n{_shorten_traceback(result.stderr)}")

    return "\n".join(parts)


class ReflexionStrategy:
    """
    M2 — Reflexion Strategy (Chiến lược Tự Phản Tỉnh)
    ===================================================
    Cơ chế: LLM tự gỡ lỗi qua tối đa N vòng lặp (Act → Execute → Reflect).
    Không cập nhật trọng số (inference-time only).
    Tín hiệu đúng/sai của compiler được dịch thành 'gradient bằng lời' (verbal gradient).
    """

    name = "reflexion"  # Nhận dạng chiến lược, dùng để lưu vào thư mục results/reflexion/

    def __init__(
        self,
        *,
        temperature: float = 0.2,      # Độ sáng tạo thấp -> sinh code ổn định, có thể tái lập
        max_tokens: int = 1024,         # Giới hạn token đầu ra mỗi lần gọi AI
        max_iterations: int = 4,        # Số vòng lặp tối đa (paper: lợi ích bão hòa vòng 3-4)
        base_url: str | None = None,    # URL server Ollama (mặc định: 192.168.31.16:11434)
        model: str | None = None,       # Tên model LLM (mặc định: deepseek-coder:1.3b)
    ) -> None:
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_iterations = max_iterations
        self.base_url = base_url
        self.model = model

    def _call_model(self, system_prompt: str, user_prompt: str) -> str:
        """Hàm tiện ích gọi Ollama với tham số cố định, tránh lặp code."""
        return generate(
            system_prompt,
            user_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            base_url=self.base_url,
            model=self.model,
        )

    def solve(self, task: Task) -> str:
        """
        Thực thi vòng lặp Reflexion cho một bài toán MBPP.

        Luồng xử lý:
          Vòng 0   : Generate (giống Baseline) → Execute → nếu pass: return
          Vòng 1-N : Reflect → cập nhật episodic_memory → Generate (mới) → Execute → nếu pass: return
          Sau N vòng: trả về code cuối cùng (dù chưa pass)

        Returns:
            str: Đoạn code Python tốt nhất mà AI sinh ra.
        """
        # Episodic memory: danh sách các đoạn tự phê tích lũy qua các vòng
        # Mục đích: chống lặp lỗi cũ (theo spec 03-M2-reflexion.yaml)
        episodic_memory: list[str] = []

        last_code = ""
        execution_result = None

        # Metadata theo dõi số vòng (run.py sẽ đọc sau khi solve() kết thúc)
        self._rounds_to_pass: int | None = None
        self._total_rounds: int = 0
        self._pass_1st_round: bool = False
        self._internal_records: list[dict] = []
        self._final_code: str = ""

        for iteration in range(self.max_iterations):

            # ==============================================================
            # PHA 1: GENERATE (Sinh mã nguồn)
            # ==============================================================
            if iteration == 0:
                # Vòng 0: Prompt đơn giản giống Baseline (tiết kiệm thời gian)
                user_prompt = (
                    f"Problem:\n{task.text}\n\n"
                    f"Unit tests:\n" + "\n".join(task.test_list)
                )
                raw_response = self._call_model(BASE_SYSTEM_PROMPT, user_prompt)

            else:
                # Vòng 1, 2, 3: Đọc toàn bộ episodic_memory + đề bài → sinh code mới
                memory_text = "\n\n".join(
                    f"[Self-critique round {i + 1}]:\n{mem}"
                    for i, mem in enumerate(episodic_memory)
                )
                user_prompt = (
                    f"Problem:\n{task.text}\n\n"
                    f"Unit tests:\n" + "\n".join(task.test_list) + "\n\n"
                    f"Your previous self-critiques:\n{memory_text}\n\n"
                    f"Now write the corrected Python function."
                )
                raw_response = self._call_model(REGENERATE_SYSTEM_PROMPT, user_prompt)

            # Bóc tách lấy code Python thuần từ phản hồi markdown của AI
            code = extract_code(raw_response)
            last_code = code

            # ==============================================================
            # PHA 2: EXECUTE (Thực thi và đánh giá qua Sandbox M0)
            # ==============================================================
            execution_result = run_executor(code=code, task=task)

            # Khởi tạo record cho vòng này (self_critique sẽ được gán sau nếu reflect)
            record: dict = {
                "iteration": iteration,
                "generate_prompt": user_prompt,     # INPUT gửi model để sinh code
                "raw_response": raw_response,        # OUTPUT thô của model (trước extract)
                "code": code,                        # code sau khi bóc tách
                "execution": {
                    "status": execution_result.status,
                    "failed_test": execution_result.failed_test,
                    "traceback": execution_result.traceback,
                    "passed_count": execution_result.passed_count,
                    "total_count": execution_result.total_count,
                    "duration_ms": execution_result.duration_ms,
                },
                "reflect_prompt": None,              # INPUT gửi model để tự phê (nếu có)
                "self_critique": None,               # OUTPUT tự phê
            }

            # ==============================================================
            # PHA 3: CHECK + REFLECT (Kiểm tra kết quả và tự phê bình)
            # ==============================================================
            self._total_rounds = iteration + 1
            if execution_result.status == "pass":
                # Code PASSED: ghi record, dừng ngay lập tức
                self._internal_records.append(record)
                self._rounds_to_pass = iteration + 1
                if iteration == 0:
                    self._pass_1st_round = True
                self._final_code = code
                return code

            # Code FAILED: bước vào Pha Reflect để sinh đoạn tự phê
            # (Chỉ reflect nếu còn ít nhất 1 vòng Generate sau đó)
            if iteration < self.max_iterations - 1:
                feedback = _build_feedback(execution_result)

                # Prompt Reflect: đề bài + code + feedback + toàn bộ episodic_memory
                memory_section = ""
                if episodic_memory:
                    memory_text = "\n\n".join(
                        f"[Self-critique round {i + 1}]:\n{mem}"
                        for i, mem in enumerate(episodic_memory)
                    )
                    memory_section = f"Previous self-critiques:\n{memory_text}\n\n"

                reflect_user_prompt = (
                    f"Problem:\n{task.text}\n\n"
                    f"Code that failed:\n```python\n{code}\n```\n\n"
                    f"Execution feedback:\n{feedback}\n\n"
                    f"{memory_section}"
                    f"Write a short self-critique (max {MAX_REFLECT_WORDS} words): "
                    f"what went wrong and what you must change."
                )
                self_critique = self._call_model(REFLECT_SYSTEM_PROMPT, reflect_user_prompt)

                # Rút gọn đoạn tự phê nếu vượt quá giới hạn từ (chống context phình)
                words = self_critique.split()
                if len(words) > MAX_REFLECT_WORDS:
                    self_critique = " ".join(words[:MAX_REFLECT_WORDS]) + "..."

                # Ghi prompt reflect (INPUT) + self_critique (OUTPUT) vào record
                record["reflect_prompt"] = reflect_user_prompt
                record["self_critique"] = self_critique
                episodic_memory.append(self_critique)

            self._internal_records.append(record)

        # Sau max_iterations vòng vẫn không pass → trả về code cuối cùng
        self._total_rounds = self.max_iterations
        self._final_code = last_code
        return last_code