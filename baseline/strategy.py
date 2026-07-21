from __future__ import annotations

from harness.extractor import extract_code
from harness.ollama_client import generate
from harness.signature import entry_function_name
from harness.types import Task

BASE_SYSTEM_PROMPT = (
    "You are an expert Python programmer. You write correct, runnable code. "
    "Return ONLY the function inside a single ```python code block."
)


class BaselineStrategy:
    name = "baseline"

    def __init__(
        self,
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url
        self.model = model

    def _user_prompt(self, task: Task) -> str:
        tests = "\n".join(task.test_list)
        func = entry_function_name(task.test_list)
        name_line = (
            f"Write a Python function named `{func}`. "
            if func
            else "Write a Python function. "
        )
        tail = (
            f"Return only the function inside a single ```python code block, "
            f"using exactly the name `{func}`."
            if func
            else "Return only the function inside a single ```python code block."
        )
        return (
            f"{task.text}\n\n"
            f"{name_line}It must pass these tests:\n"
            f"{tests}\n\n"
            f"{tail}"
        )

    def solve(self, task: Task) -> str:
        response = generate(
            BASE_SYSTEM_PROMPT,
            self._user_prompt(task),
            self.temperature,
            self.max_tokens,
            base_url=self.base_url,
            model=self.model,
        )
        return extract_code(response)
