from __future__ import annotations

import re

PYTHON_FENCE_RE = re.compile(r"```python\s*(.*?)```", re.IGNORECASE | re.DOTALL)
ANY_FENCE_RE = re.compile(r"```\s*(.*?)```", re.DOTALL)
DEF_LINE_RE = re.compile(r"(?m)^def\s+")


def extract_code(text: str) -> str:
    python_match = PYTHON_FENCE_RE.search(text)
    if python_match:
        return python_match.group(1).strip()

    any_match = ANY_FENCE_RE.search(text)
    if any_match:
        return any_match.group(1).strip()

    def_match = DEF_LINE_RE.search(text)
    if def_match:
        return text[def_match.start() :].strip()

    return text.strip()
