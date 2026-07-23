from __future__ import annotations

# Import các hàm tiện ích từ bộ khung hạ tầng harness (M0)
from harness.executor import run as run_executor  # Hàm chạy code an toàn trong Sandbox
from harness.extractor import extract_code         # Hàm bóc tách lấy code Python thuần
from baseline.strategy import BASE_SYSTEM_PROMPT              
from harness.ollama_client import generate
from harness.types import Task                    # Định nghĩa kiểu dữ liệu bài toán

# Prompt hệ thống cho kỹ thuật Reflexion: Yêu cầu AI đọc log lỗi và tự gỡ lỗi
REFLEXION_SYSTEM_PROMPT = """You are an expert Python programmer.
Your previous code attempt failed unit tests. Analyze the error trace and write a corrected version of the Python function.
Return ONLY the executable Python code block in ```python ... ``` fences."""


class ReflexionStrategy:
    name = "reflexion" 
    """
    Mô hình M2: Reflexion Strategy (Chiến lược tự phản tỉnh)
    Cơ chế: Cho phép LLM tự sửa lỗi tối đa N vòng (max_iterations) dựa trên phản hồi của compiler.
    """

    def __init__(
        self,
        *,
        temperature: float = 0.2,     # Độ sáng tạo của model (thấp để sinh code chính xác)
        max_tokens: int = 1024,        # Giới hạn số lượng từ tối đa mỗi lần sinh code
        max_iterations: int = 4,       # Số vòng lặp tự sửa tối đa (mặc định 4 lượt)
        base_url: str | None = None,   # Đường dẫn API server Ollama (http://localhost:11434)
        model: str | None = None,      # Tên model LLM (deepseek-coder:1.3b)
    ) -> None:
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_iterations = max_iterations
        self.base_url = base_url
        self.model = model

    def solve(self, task: Task) -> str:
        """
        Thực thi thuật toán 3 pha: Act -> Evaluate -> Reflect cho một bài toán.
        """
        last_code = ""

        # Vòng lặp tối đa max_iterations lượt (ví dụ: lượt 0, 1, 2, 3)
        for iteration in range(self.max_iterations):

            # --- PHA 1: ACT / GENERATE (Sinh mã nguồn) ---
            if iteration == 0:
                # Lượt đầu tiên (Lượt 0): Dùng Prompt cơ bản để sinh code như Baseline
                user_prompt = f"Problem:\n{task.text}\n\nUnit tests:\n" + "\n".join(task.test_list)
                raw_response = generate(
                    BASE_SYSTEM_PROMPT,
                    user_prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    base_url=self.base_url,
                    model=self.model,
                )
            else:
                # Các lượt sau (Lượt 1, 2, 3): Đóng gói Code bị lỗi + Log lỗi gửi lại cho AI
                reflexion_user_prompt = (
                    f"Problem:\n{task.text}\n\n"
                    f"Previous Incorrect Code:\n```python\n{last_code}\n```\n\n"
                    f"Execution Result / Error:\n{execution_result.stderr or execution_result.traceback or execution_result.status}\n\n"
                    f"Please fix the function so all tests pass."
                )
                raw_response = generate(
                    REFLEXION_SYSTEM_PROMPT,
                    reflexion_user_prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    base_url=self.base_url,
                    model=self.model,
                )

            # Lọc bỏ lời giải thích, chỉ giữ lại đoạn code Python thuần trong ```python ... ```
            code = extract_code(raw_response)
            last_code = code

            # --- PHA 2: EVALUATE (Thực thi và đánh giá trong Sandbox) ---
            execution_result = run_executor(code=code, task=task)

            # Nếu code chạy thành công và vượt qua 100% test case -> Dừng ngay lập tức
            if execution_result.status == "PASSED":
                return code

        # Nếu sau max_iterations lượt vẫn không sửa được -> Trả về bản code cuối cùng
        return last_code