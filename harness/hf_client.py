"""
harness/hf_client.py
====================
Client gọi HuggingFace Inference Endpoint (Text Generation Inference — TGI).
Endpoint này sử dụng format Text Generation (inputs/generated_text),
KHÔNG phải OpenAI-compatible chat API.

Sử dụng thay cho ollama_client khi chạy trên HuggingFace cloud.
Giao diện generate() TƯƠNG THÍCH với ollama_client.generate() → drop-in replacement.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Any

import requests

# ---------------------------------------------------------------------------
# HẰNG SỐ MẶC ĐỊNH
# ---------------------------------------------------------------------------
DEFAULT_HF_ENDPOINT_URL = (
    "https://mwfn3m833i7t9dz7.us-east4.gcp.endpoints.huggingface.cloud"
)
DEFAULT_HF_TOKEN = ""  # Đặt qua biến môi trường HF_TOKEN
DEFAULT_HF_MODEL = "deepseek-coder:1.3b"  # Tên model để ghi vào kết quả

DEFAULT_TOP_P = 1.0
DEFAULT_SEED = 5410  # Cố định seed để kết quả TÁI LẬP được
REQUEST_TIMEOUT_SECONDS = 180
BACKOFF_SECONDS = (2, 4, 8, 16)


@dataclass(frozen=True)
class GenerationLog:
    duration_ms: int
    generated_tokens: int | None


def resolve_hf_endpoint_url(url: str | None = None) -> str:
    return (url or os.environ.get("HF_ENDPOINT_URL") or DEFAULT_HF_ENDPOINT_URL).rstrip("/")


def resolve_hf_token(token: str | None = None) -> str:
    return token or os.environ.get("HF_TOKEN") or DEFAULT_HF_TOKEN


def resolve_model(model: str | None = None) -> str:
    """Trả về tên model (dùng cho metadata kết quả, không ảnh hưởng API call)."""
    return model or os.environ.get("HF_MODEL") or DEFAULT_HF_MODEL


def _build_prompt(system_prompt: str, user_prompt: str) -> str:
    """
    Ghép system + user prompt thành 1 chuỗi text duy nhất cho TGI endpoint.
    TGI không hỗ trợ chat messages, chỉ nhận text input.
    """
    return (
        f"### System:\n{system_prompt}\n\n"
        f"### User:\n{user_prompt}\n\n"
        f"### Assistant:\n"
    )


def _request_generate(
    *,
    endpoint_url: str,
    token: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    seed: int,
) -> dict[str, Any] | list[dict[str, Any]]:
    """
    Gọi HuggingFace TGI endpoint.
    POST {endpoint_url}
    Body: {"inputs": ..., "parameters": {...}}
    Response: [{"generated_text": "..."}]
    """
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": max(temperature, 0.01),  # TGI yêu cầu temperature > 0
            "return_full_text": False,  # Chỉ trả phần sinh mới, không lặp lại prompt
        },
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    response = requests.post(
        endpoint_url,
        json=payload,
        headers=headers,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def generate(
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    *,
    base_url: str | None = None,
    model: str | None = None,
    top_p: float = DEFAULT_TOP_P,
    seed: int = DEFAULT_SEED,
) -> str:
    """
    Giao diện generate() TƯƠNG THÍCH với ollama_client.generate().
    Drop-in replacement: chỉ cần import từ harness.hf_client thay vì harness.ollama_client.

    Args:
        system_prompt: Prompt hệ thống (vai trò AI).
        user_prompt:   Prompt bài toán.
        base_url:      URL endpoint HuggingFace (override DEFAULT_HF_ENDPOINT_URL).
        model:         Tên model (chỉ dùng cho metadata, không ảnh hưởng API call).
        Các tham số còn lại giống ollama_client.generate().
    """
    resolved_url = resolve_hf_endpoint_url(base_url)
    resolved_token = resolve_hf_token()
    prompt = _build_prompt(system_prompt, user_prompt)
    last_error: Exception | None = None

    for attempt in range(len(BACKOFF_SECONDS) + 1):
        start = time.monotonic()
        try:
            data = _request_generate(
                endpoint_url=resolved_url,
                token=resolved_token,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                seed=seed,
            )
            duration_ms = int((time.monotonic() - start) * 1000)

            # TGI trả về list: [{"generated_text": "..."}]
            if isinstance(data, list) and data:
                generated_text = data[0].get("generated_text", "")
            elif isinstance(data, dict):
                generated_text = data.get("generated_text", "")
            else:
                generated_text = ""

            log = GenerationLog(
                duration_ms=duration_ms,
                generated_tokens=len(generated_text.split()) if generated_text else 0,
            )
            print(
                f"generation_ms={log.duration_ms}, "
                f"generated_tokens≈{log.generated_tokens}",
                file=sys.stderr,
            )

            return str(generated_text)

        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            # 4xx (trừ 429 rate-limit) = lỗi client, không retry
            if status_code is not None and 400 <= status_code < 500 and status_code != 429:
                raise
            last_error = exc
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_error = exc

        if attempt < len(BACKOFF_SECONDS):
            wait = BACKOFF_SECONDS[attempt]
            print(f"HF request failed (attempt {attempt + 1}), retrying in {wait}s...", file=sys.stderr)
            time.sleep(wait)

    if last_error is None:
        raise RuntimeError("HF generation failed without an error")
    raise last_error
