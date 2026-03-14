"""Centralized model registry with metadata and health checks."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from inference.local_adapter import (
    InferenceBackend,
    get_client,
    get_model_name,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelInfo:
    """Metadata for a registered model."""

    name: str
    backend: InferenceBackend
    model_id: str
    context_window: int
    parameters_billions: float
    quantization: str | None = field(default=None)
    source_url: str | None = field(default=None)


# ---------------------------------------------------------------------------
# Available models
# ---------------------------------------------------------------------------

AVAILABLE_MODELS: dict[InferenceBackend, ModelInfo] = {
    InferenceBackend.GROQ: ModelInfo(
        name="GPT-OSS 120B (Groq Cloud)",
        backend=InferenceBackend.GROQ,
        model_id="openai/gpt-oss-120b",
        context_window=131072,
        parameters_billions=120.0,
        quantization=None,
        source_url="https://console.groq.com",
    ),
    InferenceBackend.OLLAMA: ModelInfo(
        name="Qwen 3.5 4B (Ollama)",
        backend=InferenceBackend.OLLAMA,
        model_id="qwen3.5:4b",
        context_window=32768,
        parameters_billions=4.0,
        quantization="Q4_K_M",
        source_url="https://ollama.com/library/qwen3.5",
    ),
    InferenceBackend.VLLM: ModelInfo(
        name="Qwen3 4B (vLLM)",
        backend=InferenceBackend.VLLM,
        model_id="Qwen/Qwen3.5-0.8B",
        context_window=8192,
        parameters_billions=4.0,
        quantization=None,
        source_url="https://huggingface.co/Qwen/Qwen3.5-0.8B",
    ),
    InferenceBackend.SGLANG: ModelInfo(
        name="Qwen3 4B (SGLang)",
        backend=InferenceBackend.SGLANG,
        model_id="Qwen/Qwen3.5-0.8B",
        context_window=8192,
        parameters_billions=4.0,
        quantization=None,
        source_url="https://huggingface.co/Qwen/Qwen3.5-0.8B",
    ),
}


def get_model_info(backend: InferenceBackend) -> ModelInfo:
    """Return metadata for a given backend's model.

    Args:
        backend: The inference backend.

    Returns:
        ``ModelInfo`` dataclass with model details.

    Raises:
        KeyError: If the backend has no registered model.
    """
    return AVAILABLE_MODELS[backend]


def list_available_models() -> list[ModelInfo]:
    """Return a list of all registered models."""
    return list(AVAILABLE_MODELS.values())


def check_backend_health(backend: InferenceBackend) -> bool:
    """Check whether a backend is reachable by issuing a tiny completion.

    Args:
        backend: The backend to check.

    Returns:
        ``True`` if the backend responded successfully, ``False`` otherwise.
    """
    try:
        client = get_client(backend)
        model = get_model_name(backend)
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
        )
        return True
    except Exception as exc:
        logger.debug("Health check failed for %s: %s", backend.value, exc)
        return False
