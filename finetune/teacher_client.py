"""
finetune/teacher_client.py
==========================
Client gọi TEACHER model qua API bên thứ ba — chuẩn OpenAI-compatible
(/chat/completions). Tương thích hầu hết nhà cung cấp: DeepSeek, OpenAI,
Together, OpenRouter, Groq, ... chỉ cần đổi 3 biến môi trường.

Cấu hình qua env:
  TEACHER_BASE_URL   vd https://api.deepseek.com/v1  (mặc định)
  TEACHER_API_KEY    khoá API
  TEACHER_MODEL      vd deepseek-chat / gpt-4o-mini / qwen2.5-coder-32b-instruct

Không phụ thuộc SDK — chỉ dùng `requests` (đã có trong requirements).
"""

from __future__ import annotations

import os
import time
from typing import Any

import requests

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"
REQUEST_TIMEOUT = 120
BACKOFF = (2, 4, 8, 16)


def resolve_base_url(url: str | None = None) -> str:
    return (url or os.environ.get("TEACHER_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


def resolve_api_key(key: str | None = None) -> str:
    return key or os.environ.get("TEACHER_API_KEY") or ""


def resolve_model(model: str | None = None) -> str:
    return model or os.environ.get("TEACHER_MODEL") or DEFAULT_MODEL


def chat(
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> str:
    """Gọi teacher (chat completions). Trả về nội dung text của message đầu tiên."""
    url = resolve_base_url(base_url) + "/chat/completions"
    key = resolve_api_key(api_key)
    payload: dict[str, Any] = {
        "model": resolve_model(model),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    last_error: Exception | None = None
    for attempt in range(len(BACKOFF) + 1):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            return str(data["choices"][0]["message"]["content"])
        except requests.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else None
            if code is not None and 400 <= code < 500 and code != 429:
                raise  # lỗi client (sai key/model) — không retry
            last_error = exc
        except (requests.Timeout, requests.ConnectionError, KeyError, ValueError) as exc:
            last_error = exc
        if attempt < len(BACKOFF):
            time.sleep(BACKOFF[attempt])
    raise last_error or RuntimeError("teacher chat failed")
