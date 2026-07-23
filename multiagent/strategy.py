from __future__ import annotations

# Import các hàm tiện ích từ bộ khung hạ tầng harness (M0)
from harness.executor import run as run_executor  # Hàm thực thi Sandbox
from harness.extractor import extract_code         # Hàm trích xuất code Python
from harness.ollama_client import generate         # Hàm gọi API Ollama
from harness.types import Task                    # Định nghĩa kiểu dữ liệu bài toán

# Prompt vai trò 1: Programmer Agent (Chuyên lập trình và sửa code)
PROGRAMMER_SYSTEM_PROMPT = """You are an expert Python programmer.
Your goal is to write clean, correct Python functions according to the problem description and unit tests.
Return ONLY the executable Python code block in ```python ... ``` fences."""

# Prompt vai trò 2: Reviewer Agent (Chuyên gia soi lỗi, phân tích nguyên nhân & đưa lời khuyên)
REVIEWER_SYSTEM_PROMPT = """You are a Senior Python Code Reviewer.
Your role is to analyze a failed Python code attempt along with its execution error trace.
Identify the exact bug, logical flaws, or edge cases, and provide constructive feedback on how the programmer should fix it.
Be concise and clear."""


class MultiAgentStrategy:
    name = "multiagent" 
    """
    Mô hình M4: Multi-Agent Strategy (Chiến lược Đa Tác Tử)
    Cơ chế: Sử dụng 2 Tác tử (Programmer & Reviewer) giao tiếp phối hợp để sinh và duyệt code.
    """

    def __init__(
        self,
        *,
        temperature: float = 0.2,     # Độ sáng tạo của model
        max_tokens: int = 1024,        # Số từ tối đa
        max_iterations: int = 4,       # Số vòng lặp tối đa giữa 2 Agent
        base_url: str | None = None,   # URL API Ollama
        model: str | None = None,      # Tên model LLM
    ) -> None:
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_iterations = max_iterations
        self.base_url = base_url
        self.model = model

    def solve(self, task: Task) -> str:
        """
        Thực thi vòng lặp tương tác giữa Programmer Agent và Reviewer Agent.
        """
        last_code = ""

        # Vòng lặp giao tiếp giữa 2 Agent (tối đa max_iterations lượt)
        for iteration in range(self.max_iterations):

            if iteration == 0:
                # Lượt 0: Programmer Agent sinh mã nguồn khởi tạo
                user_prompt = f"Problem:\n{task.text}\n\nUnit tests:\n" + "\n".join(task.test_list)
                raw_response = generate(
                    PROGRAMMER_SYSTEM_PROMPT,
                    user_prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    base_url=self.base_url,
                    model=self.model,
                )
            else:
                # Các lượt sau (1, 2, 3): Phối hợp 2 Tác tử (Reviewer -> Programmer)

                # --- BƯỚC A: REVIEWER AGENT SOI LỖI & ĐƯA RA GÓP Ý (FEEDBACK) ---
                reviewer_prompt = (
                    f"Problem:\n{task.text}\n\n"
                    f"Failed Code:\n```python\n{last_code}\n```\n\n"
                    f"Execution Error Trace:\n{execution_result.stderr or execution_result.traceback or execution_result.status}\n\n"
                    f"Please review the code, explain the root cause of the error, and provide clear fix instructions."
                )
                review_feedback = generate(
                    REVIEWER_SYSTEM_PROMPT,
                    reviewer_prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    base_url=self.base_url,
                    model=self.model,
                )

                # --- BƯỚC B: PROGRAMMER AGENT ĐỌC FEEDBACK VÀ VIẾT LAỊ CODE MỚI ---
                programmer_prompt = (
                    f"Problem:\n{task.text}\n\n"
                    f"Your Previous Code:\n```python\n{last_code}\n```\n\n"
                    f"Senior Reviewer's Feedback:\n{review_feedback}\n\n"
                    f"Based on the reviewer's feedback, write the corrected Python code."
                )
                raw_response = generate(
                    PROGRAMMER_SYSTEM_PROMPT,
                    programmer_prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    base_url=self.base_url,
                    model=self.model,
                )

            # Lọc lấy đoạn code Python từ phản hồi của Programmer Agent
            code = extract_code(raw_response)
            last_code = code

            # Đưa code vào Sandbox Executor của M0 để kiểm tra
            execution_result = run_executor(code=code, task=task)

            # Nếu đỗ (PASSED) -> Dừng ngay lập tức và trả về kết quả
            if execution_result.status == "PASSED":
                return code

        # Nếu hết số lượt giao tiếp vẫn chưa qua -> Trả về bản code cuối cùng
        return last_code