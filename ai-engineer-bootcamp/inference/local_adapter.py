"""Unified inference adapter — swap between Cloud (Groq) and Local (Ollama, vLLM, SGLang)."""

from __future__ import annotations

import logging
import os
import time
from enum import Enum

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)


class InferenceBackend(Enum):
    """Supported inference backends."""

    GROQ = "groq"
    OLLAMA = "ollama"
    VLLM = "vllm"
    SGLANG = "sglang"


# ---------------------------------------------------------------------------
# Backend configuration maps
# ---------------------------------------------------------------------------

_BASE_URLS: dict[InferenceBackend, str] = {
    InferenceBackend.GROQ: os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
    InferenceBackend.OLLAMA: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    InferenceBackend.VLLM: os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
    InferenceBackend.SGLANG: os.getenv("SGLANG_BASE_URL", "http://localhost:30000/v1"),
}

_MODEL_NAMES: dict[InferenceBackend, str] = {
    InferenceBackend.GROQ: os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
    InferenceBackend.OLLAMA: os.getenv("OLLAMA_MODEL", "qwen3.5:4b"),
    InferenceBackend.VLLM: os.getenv("VLLM_MODEL", "Qwen/Qwen3.5-0.8B"),
    InferenceBackend.SGLANG: os.getenv("SGLANG_MODEL", "Qwen/Qwen3.5-0.8B"),
}

_API_KEYS: dict[InferenceBackend, str] = {
    InferenceBackend.GROQ: os.getenv("GROQ_API_KEY", ""),
    InferenceBackend.OLLAMA: "ollama",  # Ollama doesn't require a real key
    InferenceBackend.VLLM: "EMPTY",
    InferenceBackend.SGLANG: "EMPTY",
}


def _get_default_backend() -> InferenceBackend:
    """Read the default backend from the DEFAULT_BACKEND env var."""
    raw = os.getenv("DEFAULT_BACKEND", "groq").lower()
    try:
        return InferenceBackend(raw)
    except ValueError:
        logger.warning("Unknown DEFAULT_BACKEND '%s', falling back to groq.", raw)
        return InferenceBackend.GROQ


DEFAULT_BACKEND = _get_default_backend()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_client(backend: InferenceBackend | None = None) -> OpenAI:
    """Return an OpenAI-compatible client configured for the given backend.

    Args:
        backend: The inference backend to use. Defaults to DEFAULT_BACKEND.

    Returns:
        An ``openai.OpenAI`` client pointed at the correct base URL.
    """
    backend = backend or DEFAULT_BACKEND
    api_key = _API_KEYS[backend]
    if backend == InferenceBackend.GROQ and not api_key:
        raise ValueError("GROQ_API_KEY is not set. Export it or add it to your .env file.")

    return OpenAI(base_url=_BASE_URLS[backend], api_key=api_key)


def get_model_name(backend: InferenceBackend | None = None) -> str:
    """Return the model identifier for the given backend.

    Args:
        backend: The inference backend. Defaults to DEFAULT_BACKEND.

    Returns:
        Model name string suitable for the ``model`` parameter.
    """
    backend = backend or DEFAULT_BACKEND
    return _MODEL_NAMES[backend]


def chat(
    backend: InferenceBackend | None = None,
    messages: list[dict] | None = None,
    **kwargs,
) -> str:
    """High-level chat wrapper — returns the assistant's text response.

    Args:
        backend: Inference backend to use.
        messages: List of chat messages (role/content dicts).
        **kwargs: Extra params forwarded to ``chat.completions.create``.

    Returns:
        The text content of the assistant's reply.
    """
    backend = backend or DEFAULT_BACKEND
    messages = messages or []
    client = get_client(backend)
    model = get_model_name(backend)

    logger.info("chat | backend=%s | model=%s | messages=%d", backend.value, model, len(messages))

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs,
        )
    except Exception:
        logger.exception("chat call failed | backend=%s | model=%s", backend.value, model)
        raise

    content = response.choices[0].message.content or ""
    return content.strip()


def chat_with_metrics(
    backend: InferenceBackend | None = None,
    messages: list[dict] | None = None,
    **kwargs,
) -> dict:
    """Chat with streaming to measure TTFT and throughput.

    Args:
        backend: Inference backend to use.
        messages: List of chat messages.
        **kwargs: Extra params forwarded to the completion call.

    Returns:
        Dict with keys: ``content``, ``ttft_ms``, ``total_time_ms``,
        ``prompt_tokens``, ``completion_tokens``, ``tokens_per_second``.
    """
    backend = backend or DEFAULT_BACKEND
    messages = messages or []
    client = get_client(backend)
    model = get_model_name(backend)

    logger.info(
        "chat_with_metrics | backend=%s | model=%s | messages=%d",
        backend.value,
        model,
        len(messages),
    )

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            **kwargs,
        )
    except Exception:
        logger.exception(
            "chat_with_metrics call failed | backend=%s | model=%s",
            backend.value,
            model,
        )
        raise

    start = time.perf_counter()
    first_token_time: float | None = None
    full_response = ""
    completion_tokens = 0

    for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            full_response += delta.content
            completion_tokens += 1  # Approximation: 1 chunk ~ 1 token

    total_time = time.perf_counter() - start
    ttft = (first_token_time - start) if first_token_time else total_time

    # Throughput: tokens generated / generation time (excluding TTFT)
    generation_time = total_time - ttft
    tokens_per_second = (
        completion_tokens / generation_time if generation_time > 0 else 0.0
    )

    return {
        "content": full_response.strip(),
        "ttft_ms": round(ttft * 1000, 2),
        "total_time_ms": round(total_time * 1000, 2),
        "prompt_tokens": len(str(messages)),  # Approximation for streaming
        "completion_tokens": completion_tokens,
        "tokens_per_second": round(tokens_per_second, 2),
    }
