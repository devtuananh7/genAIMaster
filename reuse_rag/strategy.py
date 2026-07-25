"""
reuse_rag/strategy.py
=====================
Chiến lược sinh code có điều kiện theo context (RAG pipeline, KHÔNG phải agent).

Luồng solve() cho một RepoTask ở một mức representation cố định:

    oracle retrieve (nạp thẳng chunk API đích theo fqn)   ← retrieval ĐÓNG BĂNG
        → render_context(chunk, level)                     ← BIẾN ĐỘC LẬP
        → build prompt (chỉ dẫn GIỐNG NHAU ở mọi mức)
        → generate (LLM)   → extract_code

Instruction "hãy dùng lại API dưới đây" giữ NGUYÊN ở mọi mức; chỉ độ giàu của
`context` thay đổi → cô lập đúng biến cần đo (reuse do context, không do lời dặn).
"""

from __future__ import annotations

import re
from typing import Any, Callable

from harness.extractor import extract_code
from harness.hf_client import generate as hf_generate
from reuse_rag.indexer import load_index
from reuse_rag.render import context_token_estimate, render_context
from reuse_rag.tasks import RepoTask

# Kiểu hàm sinh: (system, user, temperature, max_tokens, *, base_url, model) -> str
GenerateFn = Callable[..., str]

REUSE_SYSTEM_PROMPT = (
    "You are an expert Python programmer working inside an existing project. "
    "When the project already provides an API for the task, REUSE it by importing "
    "it from its module instead of reimplementing the logic yourself. "
    "Return ONLY the function inside a single ```python code block."
)


class ReuseRagStrategy:
    """Sinh code cho RepoTask ở một mức context (level) cho trước."""

    name = "reuse_rag"

    def __init__(
        self,
        *,
        level: int,
        index_path: str = "data/boltons_index.json",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        base_url: str | None = None,
        model: str | None = None,
        generate_fn: GenerateFn | None = None,
    ) -> None:
        self.level = level
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url
        self.model = model
        self._generate: GenerateFn = generate_fn or hf_generate
        self._index: dict[str, dict[str, Any]] = load_index(index_path)

    # -- prompt building --------------------------------------------------
    def build_user_prompt(self, task: RepoTask, context: str) -> str:
        # L0 (no-RAG baseline): không nhắc gì tới API project.
        if not context.strip():
            return (
                f"{task.text}\n\n"
                f"Write a function named `{task.wrapper}` that solves the task. "
                f"Return only the function inside a single ```python code block."
            )
        return (
            f"{task.text}\n\n"
            f"You may use this existing project API:\n\n"
            f"{context}\n"
            f"Write a function named `{task.wrapper}` that solves the task. "
            f"If the API above fits, import and call it. "
            f"Return only the function inside a single ```python code block."
        )

    def context_for(self, task: RepoTask) -> str:
        if self.level == 0:
            return ""  # baseline no-RAG
        chunk = self._index.get(task.target_fqn)
        if chunk is None:
            raise KeyError(
                f"target API {task.target_fqn!r} not found in index — rebuild data/boltons_index.json"
            )
        return render_context(chunk, self.level)

    # -- main -------------------------------------------------------------
    def solve(self, task: RepoTask) -> str:
        context = self.context_for(task)
        user_prompt = self.build_user_prompt(task, context)

        raw = self._generate(
            REUSE_SYSTEM_PROMPT,
            user_prompt,
            self.temperature,
            self.max_tokens,
            base_url=self.base_url,
            model=self.model,
        )
        code = extract_code(raw)

        # Metadata cho driver đọc sau khi solve()
        self._context = context
        self._context_tokens = context_token_estimate(context)
        self._raw_response = raw
        self._final_code = code
        return code


# ---------------------------------------------------------------------------
# MOCK generator — CHỈ để tự kiểm thử plumbing khi CHƯA có HF_TOKEN.
# KHÔNG phải model thật; số liệu sinh ra KHÔNG dùng làm kết quả nghiên cứu.
# Hành vi: context càng giàu (có docstring/example/body) thì càng "chịu" reuse,
# đủ để xác minh reuse_scorer + đường cong + ghi file hoạt động đầu-cuối.
# ---------------------------------------------------------------------------
def make_mock_generate() -> GenerateFn:
    api_re = re.compile(r"# Project API:\s*([\w.]+)")
    wrapper_re = re.compile(r"named `([^`]+)`")

    def mock_generate(system: str, user: str, temperature: float, max_tokens: int, **kw: Any) -> str:
        api_match = api_re.search(user)
        wrapper_match = wrapper_re.search(user)
        wrapper = wrapper_match.group(1) if wrapper_match else "solve"
        # "Giàu" nếu context có docstring/example/body (L2+).
        rich = ('"""' in user) or ("# Example:" in user) or ("# Full implementation:" in user)

        if api_match and rich:
            fqn = api_match.group(1)
            module, name = fqn.rsplit(".", 1)
            return (
                f"```python\n"
                f"from {module} import {name}\n\n"
                f"def {wrapper}(*args, **kwargs):\n"
                f"    return {name}(*args, **kwargs)\n"
                f"```"
            )
        # L1 (nghèo): giả lập model tự viết lại, KHÔNG reuse.
        return (
            f"```python\n"
            f"def {wrapper}(*args, **kwargs):\n"
            f"    raise NotImplementedError\n"
            f"```"
        )

    return mock_generate
