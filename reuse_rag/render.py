"""
reuse_rag/render.py
===================
"Input representation" — trái tim của thí nghiệm.

Cùng MỘT chunk API, render ra 4 mức độ giàu context TĂNG DẦN và LỒNG NHAU
(mỗi mức = mức trước + thêm một lớp thông tin), nên số token đơn điệu tăng:

  L1 signature   : đường dẫn import + dòng chữ ký
  L2 + docstring : L1 + mô tả (prose, đã bỏ doctest)
  L3 + example   : L2 + một ví dụ gọi cụ thể
  L4 + body      : L3 + toàn bộ mã nguồn của hàm

Dòng "# Project API: <fqn>" luôn xuất hiện ở MỌI mức: không có nó thì model
không biết import API từ đâu (reuse bất khả thi) — nên đây là thông tin nền
chung, không phải biến. Biến là docstring / example / body.
"""

from __future__ import annotations

from typing import Any

LEVEL_NAMES = {
    0: "L0_no_api",       # baseline: KHÔNG đưa API nào (điều kiện đối chứng no-RAG)
    1: "L1_signature",
    2: "L2_docstring",
    3: "L3_example",
    4: "L4_body",
}
LEVELS = (0, 1, 2, 3, 4)


def _import_hint(chunk: dict[str, Any]) -> str:
    return f"from {chunk['module']} import {chunk['name']}"


def render_context(chunk: dict[str, Any], level: int) -> str:
    """Render context cho một chunk ở mức `level` (1..4). Lồng nhau, tăng dần."""
    if level not in LEVEL_NAMES:
        raise ValueError(f"level must be one of {LEVELS}, got {level!r}")

    # L0: baseline no-RAG — không cung cấp API nào.
    if level == 0:
        return ""

    lines: list[str] = []
    # Nền chung ở mọi mức: định vị API + cách import + chữ ký.
    lines.append(f"# Project API: {chunk['fqn']}")
    lines.append(f"# Import as: {_import_hint(chunk)}")
    lines.append(chunk["signature"])

    # L2+: docstring prose.
    if level >= 2 and chunk.get("docstring"):
        doc = chunk["docstring"].strip()
        lines.append(f'    """{doc}"""')

    # L3+: một ví dụ gọi cụ thể (input→output), thụt lề từng dòng.
    if level >= 3 and chunk.get("example"):
        lines.append("    # Example:")
        for ex_line in chunk["example"].splitlines():
            lines.append(f"    {ex_line}")

    # L4: toàn bộ thân hàm (nếu có body sẽ thay thế phần khung ở trên bằng
    # nguyên bản source, nhưng vẫn giữ header import/fqn ở đầu để nhất quán).
    if level >= 4 and chunk.get("body"):
        header = [
            f"# Project API: {chunk['fqn']}",
            f"# Import as: {_import_hint(chunk)}",
            "# Full implementation:",
        ]
        return "\n".join(header) + "\n" + chunk["body"].strip() + "\n"

    return "\n".join(lines) + "\n"


def context_token_estimate(text: str) -> int:
    """Ước lượng token thô (đếm theo whitespace) — dùng cho trục chi phí."""
    return len(text.split())
