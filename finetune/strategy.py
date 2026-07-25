"""
Strategy đánh giá cho M5 Fine-tune.

Implement giao diện Strategy (solve(task) -> code) để cắm vào M0 Runner.
Load model deepseek-coder-1.3b-base trực tiếp (local inference) thay vì qua Ollama,
vì cần hỗ trợ loading LoRA adapter.

Hỗ trợ 2 backend:
  - mlx:          Mac Apple Silicon
  - transformers: PC RTX 4060

Usage trong evaluate.py hoặc qua harness.run:
    python -m harness.run --strategy finetune --tasks data/selected_tasks.json
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from harness.extractor import extract_code
from harness.signature import entry_function_name
from harness.types import Task

# ── Prompt Template ──────────────────────────────────────────────────────────
# Dùng cùng format như training data (format_training.py) để nhất quán.

_INSTRUCTION_PROMPT = """\
### Instruction:
{problem}

Write a Python function named `{func_name}`.

### Response:
```python
"""

# Completion-style prompt cho base model (trước fine-tune)
_COMPLETION_PROMPT = """\
# {problem}
# Function name: {func_name}
# Tests:
{tests_comment}

def {func_name}("""


class FinetuneStrategy:
    """Strategy đánh giá deepseek-coder-1.3b-base trước/sau fine-tune."""

    name = "finetune"

    def __init__(
        self,
        *,
        backend: str = "mlx",
        model_path: str = "deepseek-ai/deepseek-coder-1.3b-base",
        adapter_path: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        prompt_style: str = "instruction",
        # Unused but accepted for compatibility with harness.run.load_strategy
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self.backend = backend
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prompt_style = prompt_style

        self._model: Any = None
        self._tokenizer: Any = None
        self._loaded = False

    def _lazy_load(self) -> None:
        """Load model + adapter lần đầu tiên gọi solve()."""
        if self._loaded:
            return

        if self.backend == "mlx":
            self._load_mlx()
        elif self.backend == "transformers":
            self._load_transformers()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        label = "with LoRA" if self.adapter_path else "base (no adapter)"
        print(f"  [FinetuneStrategy] Loaded {self.model_path} ({label})", file=sys.stderr)
        self._loaded = True

    def _load_mlx(self) -> None:
        """Load model bằng mlx_lm (Mac Apple Silicon)."""
        try:
            from mlx_lm import load as mlx_load
        except ImportError:
            print("ERROR: pip install mlx mlx-lm", file=sys.stderr)
            sys.exit(1)

        self._model, self._tokenizer = mlx_load(
            self.model_path,
            adapter_path=self.adapter_path,
        )

    def _load_transformers(self) -> None:
        """Load model bằng transformers (PC RTX 4060)."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            print("ERROR: pip install transformers torch", file=sys.stderr)
            sys.exit(1)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )

        if self.adapter_path:
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(self._model, self.adapter_path)

        self._model.eval()

    def _build_prompt(self, task: Task) -> str:
        """Dựng prompt tùy theo style."""
        func_name = entry_function_name(task.test_list) or "solution"

        if self.prompt_style == "instruction":
            return _INSTRUCTION_PROMPT.format(
                problem=task.text.strip(),
                func_name=func_name,
            )
        else:
            # completion style
            tests_lines = "\n".join(f"# {t}" for t in task.test_list[:2])
            return _COMPLETION_PROMPT.format(
                problem=task.text.strip(),
                func_name=func_name,
                tests_comment=tests_lines,
            )

    def _generate_mlx(self, prompt: str) -> str:
        """Generate bằng mlx_lm."""
        from mlx_lm import generate as mlx_generate
        return mlx_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
            temp=self.temperature,
        )

    def _generate_transformers(self, prompt: str) -> str:
        """Generate bằng transformers."""
        import torch
        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                top_p=1.0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Chỉ lấy phần sinh mới (bỏ prompt)
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True)

    def solve(self, task: Task) -> str:
        """Sinh code cho task — interface của Strategy protocol."""
        self._lazy_load()

        prompt = self._build_prompt(task)

        if self.backend == "mlx":
            raw = self._generate_mlx(prompt)
        else:
            raw = self._generate_transformers(prompt)

        # Nếu dùng instruction prompt, response bắt đầu sau "```python\n"
        # nên raw chính là phần code. Nhưng model có thể thêm ``` ở cuối.
        # extract_code sẽ xử lý.

        # Nếu dùng completion prompt, raw bắt đầu từ phần sau "def func("
        # nên cần ghép lại.
        if self.prompt_style == "completion":
            func_name = entry_function_name(task.test_list) or "solution"
            raw = f"def {func_name}({raw}"

        return extract_code(raw)
