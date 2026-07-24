"""
multiagent/strategy.py
======================
Mô-đun M4 — Multi-Agent: Programmer <-> Reviewer (ChatDev thu nhỏ)
-------------------------------------------------------------------
Câu hỏi nghiên cứu: Feedback từ 1 agent Reviewer riêng có tốt hơn
                    tự phản tỉnh (M2 Reflexion) không?

Nguồn gốc ý tưởng: ChatDev (Qian et al., 2023) — dual-agent communication
Phần tự viết      : Reviewer agent + tích hợp trên vòng lặp M2

Thiết kế 2 tác tử (theo 05-M4-multiagent.yaml):
  ┌────────────────────────────────────────────────────────┐
  │  Agent 1: PROGRAMMER                                   │
  │  - Nhận: đề bài + nhận xét của Reviewer (vòng trước)  │
  │  - Trả về: code Python mới                            │
  └────────────────────────────────────────────────────────┘
              ↕ (trao đổi qua vòng lặp)
  ┌────────────────────────────────────────────────────────┐
  │  Agent 2: REVIEWER                                     │
  │  - Nhận: đề bài + code lỗi + ExecutionResult          │
  │  - Trả về: nhận xét có cấu trúc (KHÔNG viết code)     │
  │  - Đóng vai: Senior Reviewer khó tính                  │
  └────────────────────────────────────────────────────────┘

Ràng buộc công bằng khi so sánh với M2:
  - Cùng max_iterations = 4
  - Cùng temperature, max_tokens, model
  - Mỗi vòng M4 gọi AI 2 lần (Reviewer + Programmer) thay vì 2 lần (Reflect + Generate)
"""

from __future__ import annotations

from baseline.strategy import BASE_SYSTEM_PROMPT  # Prompt gốc dùng chung với M1, M2
from harness.executor import run as run_executor   # Sandbox Executor của M0
from harness.extractor import extract_code         # Bóc tách code Python khỏi văn xuôi AI
from harness.ollama_client import generate         # Hàm gọi POST /api/chat tới Ollama
from harness.types import Task                     # Kiểu dữ liệu bài toán MBPP

# ---------------------------------------------------------------------------
# CÁC HẰNG SỐ (cùng tiêu chuẩn với M2 để so sánh công bằng)
# ---------------------------------------------------------------------------
MAX_TRACEBACK_LINES = 15   # Giới hạn dòng traceback (chống context phình, giống M2)

# ---------------------------------------------------------------------------
# SYSTEM PROMPTS CHO TỪNG TÁC TỬ
# ---------------------------------------------------------------------------

# Prompt Reviewer: đóng vai Senior Reviewer khó tính
# Ràng buộc cứng trong prompt: KHÔNG được viết code hộ (chỉ nhận xét)
REVIEWER_SYSTEM_PROMPT = """You are a strict Senior Python Code Reviewer.
Your job is to review failed code and provide structured, actionable feedback.

Your review MUST follow this structure:
1. ERROR LOCATION: Exactly which line or logic is wrong.
2. ROOT CAUSE: Why it fails (wrong algorithm, off-by-one, wrong data type, etc.).
3. FIX DIRECTION: A clear hint on how to fix it (describe the approach, NOT the code).

HARD CONSTRAINT: You are NOT allowed to write any Python code.
Only provide textual analysis and suggestions."""

# Prompt Programmer (vòng sửa): đọc review và viết lại code
PROGRAMMER_FIX_SYSTEM_PROMPT = """You are an expert Python programmer.
You have received a detailed code review from a Senior Reviewer.
Based on the review feedback, write a corrected Python function that passes all unit tests.
Return ONLY the executable Python code block in ```python ... ``` fences."""


def _shorten_traceback(stderr: str) -> str:
    """
    Rút gọn traceback xuống MAX_TRACEBACK_LINES dòng cuối (giống M2).
    Tránh prompt phình quá lớn sau nhiều vòng lặp.
    """
    lines = stderr.strip().splitlines()
    if len(lines) <= MAX_TRACEBACK_LINES:
        return "\n".join(lines)
    return "...(truncated)...\n" + "\n".join(lines[-MAX_TRACEBACK_LINES:])


def _build_execution_context(result) -> str:
    """
    Đóng gói thông tin ExecutionResult thành đoạn văn bản rõ ràng gửi cho Reviewer.
    Reviewer nhận: loại lỗi + failed_test + traceback rút gọn.
    """
    parts = []

    # Loại lỗi tổng quát
    parts.append(f"Execution status: {result.status}")

    # Câu lệnh assert cụ thể bị rớt (giúp Reviewer biết chính xác test case nào sai)
    if getattr(result, "failed_test", None):
        parts.append(f"Failed test case: {result.failed_test}")

    # Traceback rút gọn
    if getattr(result, "traceback", None):
        parts.append(f"Traceback:\n{_shorten_traceback(result.traceback)}")
    elif getattr(result, "stderr", None):
        parts.append(f"Stderr:\n{_shorten_traceback(result.stderr)}")

    return "\n".join(parts)


class MultiAgentStrategy:
    """
    M4 — Multi-Agent Strategy (Chiến lược Đa Tác Tử)
    ==================================================
    Cơ chế: 2 Agent (Programmer + Reviewer) giao tiếp trong vòng lặp.
    Câu hỏi nghiên cứu: External critique (Reviewer) có tốt hơn
                        self-critique (M2 Reflexion) không?

    Điểm khác biệt chính so với M2:
      - M2: AI tự phê bình chính mình (self-critique)
      - M4: AI thứ 2 (Reviewer) đóng vai người ngoài phê bình (external critique)
      - Ràng buộc: Reviewer KHÔNG được viết code (chỉ nhận xét) để ép AI
                  phân tích thuần túy, không bị "kéo" vào việc sửa code
    """

    name = "multiagent"  # Nhận dạng chiến lược, lưu vào thư mục results/multiagent/

    def __init__(
        self,
        *,
        temperature: float = 0.2,      # Cùng nhiệt độ với M2 để so sánh công bằng
        max_tokens: int = 1024,         # Cùng giới hạn token với M2
        max_iterations: int = 4,        # Cùng ngân sách 4 vòng với M2 (fair comparison)
        base_url: str | None = None,    # URL server Ollama
        model: str | None = None,       # Tên model LLM (deepseek-coder:1.3b)
        reviewer_sees_execution: bool = True,  # Cấu hình nâng cao: Reviewer có thấy ExecutionResult?
    ) -> None:
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_iterations = max_iterations
        self.base_url = base_url
        self.model = model
        # reviewer_sees_execution = True  : Reviewer nhận code + ExecutionResult (mặc định)
        # reviewer_sees_execution = False : Reviewer chỉ đọc code, không thấy lỗi runtime
        #   → dùng cho thí nghiệm nâng cao (advanced_config trong 05-M4-multiagent.yaml)
        #   → tách bạch: giá trị của "người thứ 2" vs giá trị của "thông tin thực thi"
        self.reviewer_sees_execution = reviewer_sees_execution

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

    def _reviewer_critique(self, task: Task, code: str, result) -> str:
        """
        Gọi Reviewer Agent để sinh ra nhận xét có cấu trúc về đoạn code bị lỗi.

        Tuân thủ ràng buộc cứng: Reviewer KHÔNG viết code hộ (được ép trong system prompt).

        Args:
            task   : Bài toán MBPP hiện tại.
            code   : Đoạn code bị lỗi của Programmer.
            result : ExecutionResult từ Sandbox M0.

        Returns:
            str: Nhận xét có cấu trúc (Error Location + Root Cause + Fix Direction).
        """
        # Phần thông tin thực thi (tùy cấu hình reviewer_sees_execution)
        if self.reviewer_sees_execution:
            execution_context = _build_execution_context(result)
        else:
            # Cấu hình nâng cao: Reviewer tự suy luận lỗi chỉ bằng đọc code
            # Giúp tách bạch giá trị của "đôi mắt thứ hai" vs "thông tin compiler"
            execution_context = f"The code failed (execution details hidden). Status: {result.status}"

        reviewer_prompt = (
            f"Problem description:\n{task.text}\n\n"
            f"Unit tests that must pass:\n" + "\n".join(task.test_list) + "\n\n"
            f"Failed code to review:\n```python\n{code}\n```\n\n"
            f"Execution result:\n{execution_context}\n\n"
            f"Please provide your structured code review."
        )
        return self._call_model(REVIEWER_SYSTEM_PROMPT, reviewer_prompt)

    def _programmer_fix(self, task: Task, code: str, review_feedback: str) -> str:
        """
        Gọi Programmer Agent để viết lại code dựa trên nhận xét của Reviewer.

        Args:
            task           : Bài toán MBPP hiện tại.
            code           : Đoạn code cũ bị lỗi.
            review_feedback: Nhận xét có cấu trúc từ Reviewer Agent.

        Returns:
            str: Đoạn code Python mới đã được sửa (raw response, chưa extract).
        """
        programmer_prompt = (
            f"Problem:\n{task.text}\n\n"
            f"Unit tests:\n" + "\n".join(task.test_list) + "\n\n"
            f"Your previous code that failed:\n```python\n{code}\n```\n\n"
            f"Senior Reviewer's feedback:\n{review_feedback}\n\n"
            f"Based on the reviewer's structured feedback, write the corrected Python function."
        )
        return self._call_model(PROGRAMMER_FIX_SYSTEM_PROMPT, programmer_prompt)

    def solve(self, task: Task) -> str:
        """
        Thực thi vòng lặp dual-agent cho một bài toán MBPP.

        Luồng xử lý:
          Vòng 0   : Programmer sinh code (giống Baseline) → Execute → nếu pass: return
          Vòng 1-N : Reviewer nhận xét → Programmer sửa code → Execute → nếu pass: return
          Sau N vòng: trả về code cuối cùng

        Mỗi vòng (trừ vòng 0) gồm 2 lần gọi AI:
          Lần 1: Reviewer Agent (phân tích lỗi, không viết code)
          Lần 2: Programmer Agent (đọc nhận xét, viết lại code)

        Returns:
            str: Đoạn code Python tốt nhất sau các vòng trao đổi.
        """
        last_code = ""
        execution_result = None

        for iteration in range(self.max_iterations):

            # ==============================================================
            # BƯỚC A: PROGRAMMER SINH CODE
            # ==============================================================
            if iteration == 0:
                # Vòng 0: Programmer sinh code ban đầu (giống hệt Baseline M1)
                user_prompt = (
                    f"Problem:\n{task.text}\n\n"
                    f"Unit tests:\n" + "\n".join(task.test_list)
                )
                raw_response = self._call_model(BASE_SYSTEM_PROMPT, user_prompt)

            else:
                # Vòng 1, 2, 3: Programmer đọc nhận xét của Reviewer và sửa lại code
                # (raw_response đã được tính trong phần Reviewer bên dưới)
                pass  # raw_response đã được gán từ cuối vòng trước

            # Bóc tách code Python từ phản hồi markdown của Programmer
            code = extract_code(raw_response)
            last_code = code

            # ==============================================================
            # BƯỚC B: EXECUTE (Sandbox M0 chạy thử code + unit tests)
            # ==============================================================
            execution_result = run_executor(code=code, task=task)

            # ==============================================================
            # BƯỚC C: CHECK (Kiểm tra kết quả)
            # ==============================================================
            if execution_result.status == "pass":
                # Code PASSED: kết thúc vòng lặp, trả về code thành công
                return code

            # Code FAILED: kích hoạt Reviewer Agent (nếu còn vòng lặp tiếp theo)
            if iteration < self.max_iterations - 1:

                # ==============================================================
                # BƯỚC D: REVIEWER AGENT PHÂN TÍCH LỖI (External Critique)
                # ==============================================================
                # Reviewer đọc: đề bài + code lỗi + ExecutionResult
                # Reviewer KHÔNG được viết code (ép trong system prompt)
                # Reviewer trả về: nhận xét có cấu trúc (Error Location + Root Cause + Fix Direction)
                review_feedback = self._reviewer_critique(task, code, execution_result)

                # ==============================================================
                # BƯỚC E: PROGRAMMER ĐỌC NHẬN XÉT → CHUẨN BỊ SINH CODE MỚI
                # ==============================================================
                # raw_response sẽ được dùng ở đầu vòng lặp iteration+1
                raw_response = self._programmer_fix(task, code, review_feedback)

        # Sau max_iterations vòng vẫn không pass → trả về code cuối cùng
        return last_code