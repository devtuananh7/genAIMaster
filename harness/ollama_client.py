from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Any

import requests

DEFAULT_BASE_URL = "http://192.168.31.16:11434"
DEFAULT_MODEL = "deepseek-coder:1.3b"
DEFAULT_TOP_P = 1.0
DEFAULT_SEED = 5410                         # cố định để kết quả TÁI LẬP được (ablation công bằng)
REQUEST_TIMEOUT_SECONDS = 120
BACKOFF_SECONDS = (2, 4, 8)


@dataclass(frozen=True)
class GenerationLog:
    duration_ms: int
    eval_count: int | None
    prompt_eval_count: int | None


def resolve_base_url(base_url: str | None = None) -> str:
    return (base_url or os.environ.get("OLLAMA_HOST") or DEFAULT_BASE_URL).rstrip("/")


def resolve_model(model: str | None = None) -> str:
    return model or os.environ.get("OLLAMA_MODEL") or DEFAULT_MODEL


def _request_chat(
    *,
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    seed: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "top_p": top_p,
            "seed": seed,
        },
    }
    response = requests.post(
        f"{base_url}/api/chat",
        json=payload,
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
    resolved_base_url = resolve_base_url(base_url)
    resolved_model = resolve_model(model)
    last_error: Exception | None = None

    for attempt in range(len(BACKOFF_SECONDS) + 1):
        start = time.monotonic()
        try:
            data = _request_chat(
                base_url=resolved_base_url,
                model=resolved_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                seed=seed,
            )
            duration_ms = int((time.monotonic() - start) * 1000)
            log = GenerationLog(
                duration_ms=duration_ms,
                eval_count=data.get("eval_count"),
                prompt_eval_count=data.get("prompt_eval_count"),
            )
            print(
                "generation_ms={duration_ms}, eval_count={eval_count}, "
                "prompt_eval_count={prompt_eval_count}".format(**log.__dict__),
                file=sys.stderr,
            )
            message = data.get("message", {})
            return str(message.get("content", ""))
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code is not None and 400 <= status_code < 500:
                raise
            last_error = exc
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_error = exc

        if attempt < len(BACKOFF_SECONDS):
            time.sleep(BACKOFF_SECONDS[attempt])

    if last_error is None:
        raise RuntimeError("Ollama generation failed without an error")
    raise last_error
