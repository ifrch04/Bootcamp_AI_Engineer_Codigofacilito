"""Inference module — unified backend adapter, model registry, and benchmarking."""

from inference.local_adapter import (
    InferenceBackend,
    chat,
    chat_with_metrics,
    get_client,
    get_model_name,
)
from inference.model_registry import (
    AVAILABLE_MODELS,
    ModelInfo,
    check_backend_health,
    get_model_info,
    list_available_models,
)

__all__ = [
    "InferenceBackend",
    "get_client",
    "get_model_name",
    "chat",
    "chat_with_metrics",
    "ModelInfo",
    "AVAILABLE_MODELS",
    "get_model_info",
    "list_available_models",
    "check_backend_health",
]
